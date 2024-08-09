from pydub import AudioSegment

# 打开原始音频文件
input_file = './data/mz/audio/mz_audio_long.wav'
output_file = './data/mz/audio/mz_audio_3min.wav'

# 读取音频文件
audio = AudioSegment.from_wav(input_file)

# 截取前3分钟
three_minutes = 3 * 60 * 1000  # 3分钟转换为毫秒
audio_3_min = audio[:three_minutes]

# 写入新音频文件
audio_3_min.export(output_file, format='wav')
