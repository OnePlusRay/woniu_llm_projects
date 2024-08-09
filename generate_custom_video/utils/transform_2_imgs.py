import cv2
from tqdm import tqdm
from os import path
import os

video_path = r"/data/disk4/home/yaosong/ai-project/generate_custom_video/Wav2Lip/results/result_voice.mp4"

vidcap = cv2.VideoCapture(video_path)
numberOfFrames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = vidcap.get(cv2.CAP_PROP_FPS)
print("FPS: ", fps, "Frames: ", numberOfFrames)

for frameNumber in tqdm(range(numberOfFrames)):
    _,image = vidcap.read()
    cv2.imwrite(path.join('/data/disk4/home/yaosong/ai-project/generate_custom_video/data/mz/images', str(frameNumber).zfill(4)+'.jpg'), image)