import os
import subprocess
import face_detection, torch
from models import Wav2Lip
import cv2
import numpy as np
from tqdm import tqdm
import platform
import audio
from fastapi import FastAPI, HTTPException
import uvicorn
from pydantic import BaseModel
import oss2
import uuid
from dotenv import load_dotenv
from fastapi.responses import JSONResponse
from httpx import HTTPStatusError, RequestError
import shutil

# 加载 .env 文件
load_dotenv()

app = FastAPI()

# 设置文件保存的目录
MODEL_PATH = os.getenv("MODEL_PATH")
result_dir_base = os.getenv("RESULT_DIR_BASE")
temp_dir_base = os.getenv("TEMP_DIR_BASE")

# 检查并创建文件夹
if not os.path.exists(temp_dir_base):
    os.makedirs(temp_dir_base)

if not os.path.exists(result_dir_base):
    os.makedirs(result_dir_base)


device = 'cuda' if torch.cuda.is_available() else 'cpu'
mel_step_size = 16

# 阿里云OSS配置
auth = oss2.Auth(os.getenv('ACCESS_KEY_ID'), os.getenv('ACCESS_KEY_SECRET'))
bucket = oss2.Bucket(auth, os.getenv('ENDPOINT'), os.getenv('OSS_BUCKET_NAME'))

def _load(checkpoint_path):
	"""
	- 加载并返回模型的检查点数据。如果使用的是CUDA设备(GPU),则直接加载;否则,会将模型数据映射到CPU。
	"""
	if device == 'cuda':
		checkpoint = torch.load(checkpoint_path)
	else:
		checkpoint = torch.load(checkpoint_path,
								map_location=lambda storage, loc: storage)
	return checkpoint

def load_model_once(path):
	"""
	- 一次性加载并初始化模型。使用从检查点文件读取的权重更新模型参数。
	"""
	global model
	model = Wav2Lip()
	checkpoint = _load(path)
	s = checkpoint["state_dict"]
	new_s = {}
	for k, v in s.items():
		new_s[k.replace('module.', '')] = v
	model.load_state_dict(new_s)
	model = model.to(device)
	return model.eval()

def load_detector_once(device):
	"""
	- 加载面部检测模型，用于在图像中定位人脸。
	"""
	global detector
	detector = face_detection.FaceAlignment(face_detection.LandmarksType._2D, flip_input=False, device=device)
	return detector

detector = load_detector_once(device=device)
model = load_model_once(MODEL_PATH)

def get_smoothened_boxes(boxes, T):
	"""
	- 对检测到的人脸框进行平滑处理，以减少帧与帧之间的位置变化，使人脸框更稳定。
	"""
	for i in range(len(boxes)):
		if i + T > len(boxes):
			window = boxes[len(boxes) - T:]
		else:
			window = boxes[i : i + T]
		boxes[i] = np.mean(window, axis=0)
	return boxes

def face_detect(images, face_det_batch_size=16, pads=[0, 10, 0, 0], nosmooth=False, temp_dir=None):
	"""
	- 对一批图像进行人脸检测，可以选择是否对结果应用平滑处理。
	"""
	global detector
	batch_size = face_det_batch_size
	
	while 1:
		predictions = []
		try:
			for i in tqdm(range(0, len(images), batch_size)):
				predictions.extend(detector.get_detections_for_batch(np.array(images[i:i + batch_size])))
		except RuntimeError:
			if batch_size == 1: 
				raise RuntimeError('Image too big to run face detection on GPU. Please use the --resize_factor argument')
			batch_size //= 2
			print('Recovering from OOM error; New batch size: {}'.format(batch_size))
			continue
		break

	results = []
	pady1, pady2, padx1, padx2 = pads
	for rect, image in zip(predictions, images):
		if rect is None:
			cv2.imwrite(f'{temp_dir}/faulty_frame.jpg', image) # check this frame where the face was not detected.
			raise ValueError('Face not detected! Ensure the video contains a face in all the frames.')

		y1 = max(0, rect[1] - pady1)
		y2 = min(image.shape[0], rect[3] + pady2)
		x1 = max(0, rect[0] - padx1)
		x2 = min(image.shape[1], rect[2] + padx2)
		
		results.append([x1, y1, x2, y2])

	boxes = np.array(results)
	if not nosmooth: boxes = get_smoothened_boxes(boxes, T=5)
	results = [[image[y1: y2, x1:x2], (y1, y2, x1, x2)] for image, (x1, y1, x2, y2) in zip(images, boxes)]

	return results 

def datagen(frames, mels, box=[-1, -1, -1, -1], static=False, img_size=96, wav2lip_batch_size=128, temp_dir=None):
	"""
	- 生成器函数,用于在每个批次中准备图像和对应的音频特征(Mel频谱)以供模型使用。
	"""
	img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

	if box[0] == -1:
		if not static:
			face_det_results = face_detect(frames, temp_dir=temp_dir) # BGR2RGB for CNN face detection
		else:
			face_det_results = face_detect([frames[0]], temp_dir=temp_dir)
	else:
		print('Using the specified bounding box instead of face detection...')
		y1, y2, x1, x2 = box
		face_det_results = [[f[y1: y2, x1:x2], (y1, y2, x1, x2)] for f in frames]

	for i, m in enumerate(mels):
		idx = 0 if static else i%len(frames)
		frame_to_save = frames[idx].copy()
		face, coords = face_det_results[idx].copy()

		face = cv2.resize(face, (img_size, img_size))
			
		img_batch.append(face)
		mel_batch.append(m)
		frame_batch.append(frame_to_save)
		coords_batch.append(coords)

		if len(img_batch) >= wav2lip_batch_size:
			img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

			img_masked = img_batch.copy()
			img_masked[:, img_size//2:] = 0

			img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
			mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

			yield img_batch, mel_batch, frame_batch, coords_batch
			img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

	if len(img_batch) > 0:
		img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

		img_masked = img_batch.copy()
		img_masked[:, img_size//2:] = 0

		img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
		mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

		yield img_batch, mel_batch, frame_batch, coords_batch

def wav2lip(face_path, audio_path, resize_factor=1, fps=25., rotate=False, crop=[0, -1, 0, -1], wav2lip_batch_size=128,
			outfile='results/result_voice.mp4', temp_dir=None):
	"""
	- 主要的处理函数，它整合了视频和音频输入，使用模型生成视频输出，其中人物的口型与提供的音频同步。
	"""
	global model, detector

	if not os.path.isfile(face_path):
		raise ValueError('--face argument must be a valid path to video/image file')

	elif face_path.split('.')[1] in ['jpg', 'png', 'jpeg']:
		full_frames = [cv2.imread(face_path)]
		fps = fps

	else:
		video_stream = cv2.VideoCapture(face_path)
		fps = video_stream.get(cv2.CAP_PROP_FPS)

		print('Reading video frames...')

		full_frames = []
		while 1:
			still_reading, frame = video_stream.read()
			if not still_reading:
				video_stream.release()
				break
			if resize_factor > 1:
				frame = cv2.resize(frame, (frame.shape[1]//resize_factor, frame.shape[0]//resize_factor))

			if rotate:
				frame = cv2.rotate(frame, cv2.cv2.ROTATE_90_CLOCKWISE)

			y1, y2, x1, x2 = crop
			if x2 == -1: x2 = frame.shape[1]
			if y2 == -1: y2 = frame.shape[0]

			frame = frame[y1:y2, x1:x2]

			full_frames.append(frame)

	print ("Number of frames available for inference: "+str(len(full_frames)))

	if not audio_path.endswith('.wav'):
		print('Extracting raw audio...')
		command = 'ffmpeg -y -i {} -strict -2 {}'.format(audio_path, f'{temp_dir}/temp.wav')
    
		subprocess.call(command, shell=True)
		audio_path = f'{temp_dir}/temp.wav'

	wav = audio.load_wav(audio_path, 16000)
	mel = audio.melspectrogram(wav)
	print(mel.shape)

	if np.isnan(mel.reshape(-1)).sum() > 0:
		raise ValueError('Mel contains nan! Using a TTS voice? Add a small epsilon noise to the wav file and try again')

	mel_chunks = []
	mel_idx_multiplier = 80./fps 
	i = 0
	while 1:
		start_idx = int(i * mel_idx_multiplier)
		if start_idx + mel_step_size > len(mel[0]):
			mel_chunks.append(mel[:, len(mel[0]) - mel_step_size:])
			break
		mel_chunks.append(mel[:, start_idx : start_idx + mel_step_size])
		i += 1

	print("Length of mel chunks: {}".format(len(mel_chunks)))

	full_frames = full_frames[:len(mel_chunks)]

	batch_size = wav2lip_batch_size
	gen = datagen(full_frames.copy(), mel_chunks, temp_dir=temp_dir)

	for i, (img_batch, mel_batch, frames, coords) in enumerate(tqdm(gen, 
											total=int(np.ceil(float(len(mel_chunks))/batch_size)))):
		if i == 0:
			frame_h, frame_w = full_frames[0].shape[:-1]
			out = cv2.VideoWriter(f'{temp_dir}/result.avi', 
									cv2.VideoWriter_fourcc(*'DIVX'), fps, (frame_w, frame_h))

		img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)
		mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(device)

		with torch.no_grad():
			pred = model(mel_batch, img_batch)

		pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.
		
		for p, f, c in zip(pred, frames, coords):
			y1, y2, x1, x2 = c
			p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))

			f[y1:y2, x1:x2] = p
			out.write(f)

	out.release()

	command = 'ffmpeg -y -i {} -i {} -strict -2 -q:v 1 {}'.format(audio_path, f'{temp_dir}/result.avi', outfile)
	subprocess.call(command, shell=platform.system() != 'Windows')
	
def upload_mp4_to_oss(file_path: str):
    """
    从本地路径上传MP4文件到阿里云OSS。

    参数:
        file_path (str): 本地MP4文件的路径。

    返回:
        str: OSS中文件的URL或错误信息。
    """
    if not file_path.endswith('.mp4'):
        raise ValueError("The file is not an MP4 file.")
	
    id = uuid.uuid1()

    save_path = f"generate_custom_video/results/{str(id)}/result_voice.mp4"
    file_url = f"https://{os.getenv('OSS_BUCKET_NAME')}.{os.getenv('ENDPOINT')}/generate_custom_video/results/{str(id)}/result_voice.mp4"
    
    try:
        # 打开文件并上传到OSS
        with open(file_path, 'rb') as file_data:
            bucket.put_object(save_path, file_data)
        return file_url
    except oss2.exceptions.OssError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except IOError as e:
        raise HTTPException(status_code=500, detail=f"Error opening file: {e}")
	
class SyncRequest(BaseModel):
    video_path: str
    audio_path: str
    srt_path: str = None

@app.post("/process_sync/")
async def process_sync(request: SyncRequest):
    video_path = request.video_path 
    audio_path = request.audio_path

    wav_uuid = str(uuid.uuid1())
    temp_dir = os.path.join(temp_dir_base, wav_uuid)
    result_dir = os.path.join(result_dir_base, wav_uuid)

	# 检查并创建文件夹
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    result_video_path = os.path.join(result_dir, 'result_voice.mp4')
    result_with_srt_path = os.path.join(result_dir, 'result_with_srt.mp4')

    if not os.path.exists(video_path) or not os.path.exists(audio_path):
        raise HTTPException(status_code=404, detail="File not found.")

    try:
        wav2lip(face_path=video_path, audio_path=audio_path, resize_factor=2, outfile=result_video_path, temp_dir=temp_dir)
        # 检查是否有字幕文件需要处理
        if request.srt_path:
            # 构造完整的 ffmpeg 命令，注意 subprocess.run 需要传入命令的参数列表
            ffmpeg_command = [
				'ffmpeg', '-y', '-i', result_video_path, 
				'-vf', f"subtitles={request.srt_path}:force_style='Fontname=DejaVu Serif,FontSize=16,PrimaryColour=&H00FFFFFF,Scale=0.9'",
				result_with_srt_path
			]
            # 执行 ffmpeg 命令
            subprocess.run(ffmpeg_command)
            
            # 上传处理后的视频到 OSS
            file_url = upload_mp4_to_oss(result_with_srt_path)
        else:
            # 如果没有字幕，直接上传原始合成的视频
            file_url = upload_mp4_to_oss(result_video_path)
        
        return {"file_url": file_url}  # 返回上传文件的 URL

    except HTTPStatusError as http_err:
        return JSONResponse(status_code=http_err.response.status_code, content={"message": str(http_err)})
    except RequestError as req_err:
        return JSONResponse(status_code=500, content={"message": f"Network error: {str(req_err)}"})
    except IOError as io_err:
        return JSONResponse(status_code=500, content={"message": f"File I/O error: {str(io_err)}"})
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": f"An error occurred: {str(e)}"})
    finally:
		# 清理生成的临时文件
        if os.path.exists(result_dir):
            shutil.rmtree(result_dir)
		# 删除./temp文件夹下所有文件并重建文件夹
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        if request.srt_path:
            os.remove(request.srt_path)

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000, workers=1)