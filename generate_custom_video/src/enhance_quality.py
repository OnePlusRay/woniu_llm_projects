import cv2
import matplotlib.pyplot as plt
import os
import ffmpeg
def display(img1, img2):
  fig = plt.figure(figsize=(25, 10))
  ax1 = fig.add_subplot(1, 2, 1) 
  plt.title('Input', fontsize=16)
  ax1.axis('off')
  ax2 = fig.add_subplot(1, 2, 2)
  plt.title('CodeFormer', fontsize=16)
  ax2.axis('off')
  ax1.imshow(img1)
  ax2.imshow(img2)
def imread(img_path):
  img = cv2.imread(img_path)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  return img



restoredFramesPath = r"/data/disk4/home/yaosong/ai-project/generate_custom_video/CodeFormer/results/images_0.7/final_results"
processedVideoOutputPath = r'/data/disk4/home/yaosong/ai-project/generate_custom_video/data/mz/output'

dir_list = os.listdir(restoredFramesPath)
dir_list.sort()

import cv2
import numpy as np

batch = 0
batchSize = 600
from tqdm import tqdm
for i in tqdm(range(0, len(dir_list), batchSize)):
    img_array = []
    start, end = i, i+batchSize
    print("processing ", start, end)
    for filename in  tqdm(dir_list[start:end]):
        filename = os.path.join(restoredFramesPath, filename)
        img = cv2.imread(filename)
        if img is None:
            continue
        height, width, layers = img.shape
        size = (width,height)
        img_array.append(img)


out = cv2.VideoWriter(processedVideoOutputPath+'/batch_'+str(batch).zfill(4)+'.avi',cv2.VideoWriter_fourcc(*'DIVX'), 30, size)
batch = batch + 1
 
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()

# 音视频合并 Generate audio video
audio = ffmpeg.input(f'/data/disk4/home/yaosong/ai-project/generate_custom_video/data/mz/audio/mz_audio_3min.wav')
video = ffmpeg.input(f'/data/disk4/home/yaosong/ai-project/generate_custom_video/data/mz/video/mz_video_3min.mp4')
print("合并视音频")
out = ffmpeg.output(video, audio, f'/data/disk4/home/yaosong/ai-project/generate_custom_video/data/mz/final_output/final.mp4')
out.run()  
print("恭喜您，音视频合并完成，存放在output/final.mp4")