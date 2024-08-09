from moviepy.video.io.VideoFileClip import VideoFileClip

# 打开原始视频文件
input_file = './data/mz/video/mz_video_long.mp4'
output_file = './data/mz/video/mz_video_3min.mp4'

# 创建 VideoFileClip 对象
clip = VideoFileClip(input_file)

# 截取前3分钟
clip_3_min = clip.subclip(0, 3*60)  # 3*60表示3分钟

# 写入新的视频文件
clip_3_min.write_videofile(output_file, codec='libx264', audio_codec='aac')

# 关闭 clip 对象
clip.close()
clip_3_min.close()
