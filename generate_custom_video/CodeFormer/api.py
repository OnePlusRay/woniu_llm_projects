# server.py
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel, validator
import cv2
import os
import subprocess
import shutil
import oss2
import uuid
import ffmpeg
from dotenv import load_dotenv
import uvicorn
from moviepy.editor import VideoFileClip
import GPUtil
import httpx
import logging
from urllib.parse import urlparse
import time
from multiprocessing import Pool

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 加载 .env 文件
load_dotenv()

# 阿里云OSS配置
auth = oss2.Auth(os.getenv('ACCESS_KEY_ID'), os.getenv('ACCESS_KEY_SECRET'))
bucket = oss2.Bucket(auth, os.getenv('ENDPOINT'), os.getenv('OSS_BUCKET_NAME'))

base_folder=os.getenv("BASE_FOLDER")
result_folder=os.getenv("RESULT_FOLDER")
processed_video_path=os.getenv("PROCESSED_VIDEO_PATH")
final_video_path=os.getenv("FINAL_VIDEO_PATH")
output_audio_path=os.getenv("OUTPUT_AUDIO_PATH")
local_video_path=os.getenv("LOCAL_VIDEO_PATH")

# 检查并创建文件夹
if not os.path.exists(base_folder):
    os.makedirs(base_folder)

if not os.path.exists(result_folder):
    os.makedirs(result_folder)

if not os.path.exists(processed_video_path):
    os.makedirs(processed_video_path)

if not os.path.exists(final_video_path):
    os.makedirs(final_video_path)

if not os.path.exists(output_audio_path):
    os.makedirs(output_audio_path)

if not os.path.exists(local_video_path):
    os.makedirs(local_video_path)


def read_image(image_path):
    """读取图像并返回图像对象及其尺寸。"""
    img = cv2.imread(image_path)
    if img is not None:
        height, width, layers = img.shape
        return (img, (width, height))
    else:
        raise FileNotFoundError(f"Image {image_path} not found")

def wait_for_gpu_availability(max_wait_time=600, interval=10):
    total_wait_time = 0
    while total_wait_time < max_wait_time:
        processes = get_gpu_processes()
        if processes > 0:
            return True
        time.sleep(interval)
        total_wait_time += interval
    return False

def get_gpu_processes():
    try:
        # 获取GPU信息
        gpus = GPUtil.getGPUs()
        if not gpus:
            raise RuntimeError("No GPU found")

        # 假设使用第一块GPU
        gpu = gpus[0]
        free_memory = gpu.memoryFree   # 可用显存

        # 计算可以使用的进程数
        processes_per_gpu = min(int(free_memory / 4000), 20)  # 每4GB显存允许一个进程，最多20个
        return processes_per_gpu
    except Exception as e:
        logger.error(f"获取GPU进程数失败: {str(e)}")
        raise

def split_folder_into_subfolders(source_folder, target_folder, subfolder_count):
    try:
        files = [f for f in os.listdir(source_folder) if os.path.isfile(os.path.join(source_folder, f))]
        if not os.path.exists(target_folder):
            os.makedirs(target_folder)
        files_per_folder = len(files) // subfolder_count
        remainder = len(files) % subfolder_count

        start_index = 0
        for i in range(subfolder_count):
            subfolder_path = os.path.join(target_folder, f'subfolder_{i+1}')
            if not os.path.exists(subfolder_path):
                os.makedirs(subfolder_path)
            
            # Calculate the number of files to move to this subfolder
            end_index = start_index + files_per_folder + (1 if i < remainder else 0)
            for file in files[start_index:end_index]:
                shutil.move(os.path.join(source_folder, file), subfolder_path)
            start_index = end_index
    except Exception as e:
        logger.error(f"分割子文件夹失败: {str(e)}")
        raise

def codeformer_main(base_folder, result_folder):
    try:
        subfolders = [os.path.join(base_folder, f) for f in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, f))]
        processes = []

        for subfolder in subfolders:
            proc = subprocess.Popen(['python', 'inference_codeformer.py', '-w', '0.7', '--input_path', subfolder, '--output_path', result_folder])
            processes.append(proc)

        for proc in processes:
            proc.wait()  # Wait for each process to complete
    except Exception as e:
        logger.error(f"CodeFormer多进程处理失败: {str(e)}")
        raise

def clean_up_directory(path: str):
    """
    清理指定目录下的所有文件和文件夹。

    参数:
        path (str): 需要清理的目录路径。
    """
    if os.path.exists(path):
        shutil.rmtree(path)

class VideoProcessor:
    def __init__(self, base_folder, result_folder, processed_video_path, final_video_path, output_audio_path, local_video_path, oss_bucket):
        unique_id = uuid.uuid4()
        self.base_folder = os.path.join(base_folder, str(unique_id))
        self.result_folder = os.path.join(result_folder, str(unique_id))
        self.processed_video_path = os.path.join(processed_video_path, str(unique_id), 'processed_video.mp4')
        self.final_video_path = os.path.join(final_video_path, str(unique_id), 'final_video.mp4')
        self.output_audio_path = os.path.join(output_audio_path, str(unique_id), 'output_audio.wav')
        self.local_video_path = os.path.join(local_video_path, str(unique_id), 'local_video.mp4')
        os.makedirs(self.base_folder, exist_ok=True)
        os.makedirs(self.result_folder, exist_ok=True)
        os.makedirs(os.path.dirname(self.processed_video_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.final_video_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.output_audio_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.local_video_path), exist_ok=True)
        self.oss_bucket = oss_bucket
        self.logger = logging.getLogger(self.__class__.__name__)

    def download_video(self, video_url):
        try:
            # 验证URL
            parsed_url = urlparse(video_url)
            if parsed_url.scheme not in ["http", "https"]:
                self.logger.error("只支持HTTP或HTTPS协议")
                raise HTTPException(status_code=400, detail="只支持HTTP或HTTPS协议")
            
            # 发起HTTP请求
            with httpx.stream("GET", video_url, timeout=60.0) as response:
                response.raise_for_status()  # Ensure we notice bad responses

                content_type = response.headers.get('Content-Type')
                # 限制下载的文件类型
                if not content_type or not content_type.startswith('video'):
                    raise HTTPException(status_code=400, detail="不支持的文件类型")

                # 限制下载的文件大小（例如：最大200MB）
                max_file_size = 200 * 1024 * 1024  # 200MB
                total_bytes = 0
                with open(self.local_video_path, 'wb') as output_file:
                    for chunk in response.iter_bytes(chunk_size=8192):
                        output_file.write(chunk)
                        total_bytes += len(chunk)
                        if total_bytes > max_file_size:
                            raise HTTPException(status_code=400, detail="文件大小超出限制")
        except httpx.HTTPError as e:
            self.logger.error(f"下载视频失败: {e}")
            raise HTTPException(status_code=400, detail="视频下载失败")
    
    def extract_audio_from_video(self):
        try:
            # 加载视频文件
            video = VideoFileClip(self.local_video_path)
            
            # 提取音频
            audio = video.audio
            
            # 将音频保存为WAV格式
            audio.write_audiofile(self.output_audio_path, codec='pcm_s16le')  # WAV通常使用pcm_s16le编解码器
            self.logger.info("音频提取并保存成功")
        except Exception as e:
            self.logger.error(f"提取音频失败: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="提取音频失败")
        finally:
            # 释放资源
            if 'video' in locals():
                video.close()
                self.logger.info("视频资源已释放")

    def convert_video_to_images(self):
        vidcap = None
        try:
            vidcap = cv2.VideoCapture(self.local_video_path)
            number_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.logger.info(f"准备转换视频到图片，总帧数: {number_frames}")
        
            for frameNumber in range(number_frames):
                success, image = vidcap.read()
                if not success:
                    self.logger.warning(f"在帧 {frameNumber} 处未能读取图像")
                    continue
                cv2.imwrite(os.path.join(self.base_folder, f'{frameNumber:04d}.jpg'), image)
            self.logger.info("视频转换为图片完成")
        except Exception as e:
            self.logger.error(f"视频转换为图片失败: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="视频转换为图片失败")
        finally:
            if vidcap:
                vidcap.release()
                self.logger.info("视频捕获资源已释放")

    def process_images_with_codeformer(self):
        self.logger.info("开始计算GPU可用进程")
        processes_per_gpu = get_gpu_processes()
        self.logger.info(f"GPU可用进程数: {processes_per_gpu}")
        
        self.logger.info("开始将图片分割到子文件夹")
        split_folder_into_subfolders(self.base_folder, self.base_folder, processes_per_gpu)
        self.logger.info("图片分割完成")

        self.logger.info("开始使用CodeFormer进行人脸修复")
        codeformer_main(self.base_folder, self.result_folder)
        self.logger.info("人脸修复完成")

    def merge_images_to_video(self):
        try:
            # 获取图像文件路径列表
            image_files = sorted(os.listdir(os.path.join(self.result_folder, 'final_results')))
            image_paths = [os.path.join(self.result_folder, 'final_results', filename) for filename in image_files]
            
            # 使用多进程池读取所有图像
            with Pool() as pool:
                results = pool.map(read_image, image_paths)
            
            # 检查是否有图像被处理
            if not results:
                raise ValueError("No images to process.")
            
            # 从第一个图像获取视频尺寸
            _, size = results[0]

            # 读取原视频
            cap = cv2.VideoCapture(self.local_video_path)

            # 获取原视频的帧率
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            # 创建视频写入器
            out = cv2.VideoWriter(self.processed_video_path, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
            
            # 写入图像到视频
            for img, _ in results:
                out.write(img)
        except Exception as e:
            logger.error(f"合并图片为视频失败: {str(e)}")
            raise
        finally:
            if 'out' in locals():
                out.release()
                self.logger.info("视频合成资源已释放")

    def merge_audio_and_video(self):
        try:
            audio = ffmpeg.input(self.output_audio_path)
            video = ffmpeg.input(self.processed_video_path)
            out = ffmpeg.output(video, audio, self.final_video_path)
            out.run()
        except ffmpeg.Error as e:
            logger.error(f"ffmpeg命令执行失败: {e.stderr}")
            raise
        except Exception as e:
            logger.error(f"音视频合并失败: {str(e)}")
            raise
    
    def upload_mp4_to_oss(self):
        """
        从本地路径上传MP4文件到阿里云OSS。

        参数:
            file_path (str): 本地MP4文件的路径。

        返回:
            str: OSS中文件的URL或错误信息。
        """
        if not self.final_video_path.endswith('.mp4'):
            raise ValueError("The file is not an MP4 file.")
        
        id = uuid.uuid1()

        save_path = f"generate_custom_video/results/{str(id)}/final_video.mp4"
        file_url = f"https://{os.getenv('OSS_BUCKET_NAME')}.{os.getenv('ENDPOINT')}/generate_custom_video/results/{str(id)}/final_video.mp4"
        
        try:
            # 打开文件并上传到OSS
            with open(self.final_video_path, 'rb') as file_data:
                bucket.put_object(save_path, file_data)
            return file_url
        except oss2.exceptions.OssError as e:
            raise HTTPException(status_code=500, detail=str(e))
        except IOError as e:
            raise HTTPException(status_code=500, detail=f"Error opening file: {e}")
        
    def clean_up(self):
        try:
            self.logger.info("开始清理临时文件和文件夹")
            clean_up_directory(self.base_folder)
            clean_up_directory(self.result_folder)
            clean_up_directory(os.path.dirname(self.processed_video_path))
            clean_up_directory(os.path.dirname(self.output_audio_path))
            self.logger.info("临时文件和文件夹清理完成")
        except Exception as e:
            self.logger.warning(f"清理资源时发生错误: {e}")

class VideoProcessRequest(BaseModel):
    video_path: str

    @validator('video_path')
    def validate_url(cls, value):
        parsed = urlparse(value)
        if parsed.scheme not in ['http', 'https']:
            raise ValueError('Invalid URL scheme')
        if not parsed.netloc:
            raise ValueError('Invalid URL')
        return value

app = FastAPI()

@app.post("/process_video/")
async def process_video(request: VideoProcessRequest):
    processor = VideoProcessor(
        base_folder=base_folder,
        result_folder=result_folder,
        processed_video_path=processed_video_path,
        final_video_path=final_video_path,
        output_audio_path=output_audio_path,
        local_video_path=local_video_path,
        oss_bucket=bucket
        )
    logger.info("开始视频处理请求")
    try:
        logger.info(f"开始下载视频: {request.video_path}")
        processor.download_video(video_url=request.video_path)
        logger.info("视频下载完成")

        logger.info("开始从视频提取音频")
        processor.extract_audio_from_video()

        processor.convert_video_to_images()
        logger.info("视频转换为图片完成")

        logger.info("检测是否有可用GPU")
        if not wait_for_gpu_availability():
            logger.error("暂时没有GPU资源. 请稍后重试或等待资源可用.")
        else:
            # 继续执行需要GPU的任务
            logger.info("开始使用GPU调用codeformer")

        # 多进程调用codeformer
        processor.process_images_with_codeformer()

        logger.info("开始将修复后的图片转换为视频")
        processor.merge_images_to_video()
        logger.info("图片转换为视频完成")

        logger.info("开始合并音频和视频")
        processor.merge_audio_and_video()
        logger.info("音视频合并完成")

        logger.info("开始上传视频到阿里云OSS")
        oss_url = processor.upload_mp4_to_oss()
        logger.info(f"视频上传成功。OSS URL: {oss_url}")

    except httpx.HTTPStatusError as http_err:
        # HTTP请求返回了不成功的状态码
        logger.error(f"HTTP status error occurred during video processing: {http_err}", exc_info=True)
        return JSONResponse(status_code=http_err.response.status_code, content={"message": "HTTP status error occurred during video processing"})
    except httpx.RequestError as req_err:
        # 请求过程中发生了网络问题
        logger.error(f"Request error during video processing: {req_err}", exc_info=True)
        return JSONResponse(status_code=500, content={"message": f"Network error: {req_err}"})
    except IOError as io_err:
        # 文件读写错误
        logger.error(f"File I/O error during video processing: {io_err}", exc_info=True)
        return JSONResponse(status_code=500, content={"message": f"File I/O error: {io_err}"})
    except Exception as e:
        # 其他类型错误
        logger.error(f"An error occurred during video processing: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"message": f"An error occurred: {e}"})
    finally:
        processor.clean_up()

    return {"oss_url": oss_url}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8082)