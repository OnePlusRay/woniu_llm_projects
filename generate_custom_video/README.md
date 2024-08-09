# 项目API启动流程文档
基于GPT-SoVITS, wav2Lip, CodeFormer实现自定义生成视频。**（python=3.10）**

## 目录
- [项目API启动流程文档](#项目api启动流程文档)
  - [目录](#目录)
  - [整体流程](#整体流程)
  - [GPT-SoVITS](#gpt-sovits)
    - [基本流程](#基本流程)
    - [文件结构](#文件结构)
    - [API启动服务](#api启动服务)
      - [模型准备](#模型准备)
      - [安装依赖与运行api脚本](#安装依赖与运行api脚本)
  - [Wav2Lip](#wav2lip)
    - [基本流程](#基本流程-1)
    - [文件结构](#文件结构-1)
    - [API启动服务](#api启动服务-1)
      - [模型准备](#模型准备-1)
      - [安装依赖与运行api脚本](#安装依赖与运行api脚本-1)
  - [generate\_custom\_video](#generate_custom_video)
    - [文件结构](#文件结构-2)
    - [出入参](#出入参)
    - [API启动服务](#api启动服务-2)
      - [安装依赖与运行api脚本](#安装依赖与运行api脚本-2)
      - [发起请求](#发起请求)
  - [独立Stable-Diffusion文生图+视频背景替换服务](#独立stable-diffusion文生图视频背景替换服务)
    - [基本流程](#基本流程-2)
  - [独立CodeFormer视频高清化服务](#独立codeformer视频高清化服务)
    - [API启动流程](#api启动流程)


## 整体流程
1. 数据准备：一段用于生成音频的文本，一个模板视频
2. 启动API：按照下面的流程依次启动GPT-SoVITS、Wav2Lip、generate_custom_video的API（generate_custom_video是前面两个模型的整合，**需要三个API都启动才可以正常运行**）
3. 运行服务：输入一段文本和模板视频的编号或url，运行服务
4. 处理逻辑：模型GPT-SoVITS根据输入的文本生成一个音频临时文件，再将这个音频文件和输入的模板视频传入模型Wav2Lip，根据音频文件的信息学习到对应的唇部运动，最后把这段音频和唇部运动合成到原来的模板视频中，得到一个新的视频
5. 返回结果：API接口处会返回一个存放生成视频的url

## GPT-SoVITS
### 基本流程
- 给定一段文本，生成一个音频文件

### 文件结构
- GPT_weights 和 Sovits_weights：存放模型的权重参数文件
- GPT-SoVITS/pretrained_models：存放预训练模型（需要在huggingface上下载）
- ```mz6s.wav```：原项目音频文件
- ```api.py```：主函数
- ```.env```：环境配置文件，主要包含srt字幕文件的生成路径和阿里云oss的路径配置
```shell
SRT_PATH=/data/disk4/home/chenrui/chenruitmp2/temp
```
- ```config.py```：配置文件，包含用到模型路径的配置
```python
# 推理过程用的指定模型
sovits_path = "/data/disk4/home/chenrui/ai-project/generate_custom_video/GPT-SoVITS/Sovits_weights/xxx_e8_s56.pth"
gpt_path = "/data/disk4/home/chenrui/ai-project/generate_custom_video/GPT-SoVITS/GPT_weights/xxx-e15.ckpt"
# 下面的路径是相对路径，当前目录是/data/disk4/home/chenrui/ai-project/generate_custom_video/GPT-SoVITS/
cnhubert_path = "GPT_SoVITS/pretrained_models/chinese-hubert-base"
bert_path = "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large"
pretrained_sovits_path = "GPT_SoVITS/pretrained_models/s2G488k.pth"
pretrained_gpt_path = "GPT_SoVITS/pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt"
```

### API启动服务
#### 模型准备
- 从 https://huggingface.co/lj1995/GPT-SoVITS/tree/main 下载预训练模型，并将它们放置在 GPT_SoVITS\pretrained_models 中。(使用需要安装ffmpeg)
- 把训练好的模型的权重参数文件存放在文件夹GPT_weights和Sovits_weights中。

#### 安装依赖与运行api脚本
```shell
cd GPT-SoVITS

proxychains pip install -r requirements.txt

# 需要配置config.py文件或在启动api脚本是指定路径参数
python api.py -dr "mz6s.wav" -dt "今天咱们是一个果然好物专场。秋去冬来，诚意满满，给你 不一样的购物体验。" -dl "zh"
```

接口地址-http://127.0.0.1:9880

## Wav2Lip
### 基本流程
- 准备输入的音频和视频文件。音频通常是语音文件，视频是包含人脸的文件。
- 将处理后的视频帧和音频特征输入Wav2Lip模型，生成同步的唇部运动。
- 将生成的唇部运动叠加到原始视频帧上，生成最终的视频输出。

### 文件结构
- checkpoints：存放训练好（或训练到一定程度）的模型，便于推理时直接加载
- data：包含audio和video两个子文件夹，分别存放音频和视频文件
- ```api.py```：主函数
- ```.env```：主要包括阿里云oss和wav2lip模型的路径配置，其中模型路径配置如下
```shell
MODEL_PATH=checkpoints/checkpoint_step000316000.pth
RESULT_DIR_BASE=results
TEMP_DIR_BASE=temp
```

### API启动服务
#### 模型准备
- 下载s3fd面部检测模型
```shell
proxychains wget "https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth" -O "./Wav2Lip/face_detection/detection/sfd/s3fd.pth"
```
注：若连接不上可以手动下载

- 将当前模型的.pth文件放入checkpoints文件夹中
目前的文件名是```checkpoint_step000316000.pth```

#### 安装依赖与运行api脚本
```shell
cd Wav2Lip

proxychains pip install -r requirements.txt
proxychains pip install -q youtube-dl
proxychains pip install ffmpeg-python
proxychains pip install librosa==0.9.1

python api.py
```

接口地址-http://127.0.0.1:8000/process_sync/

## generate_custom_video
### 文件结构
- ```api.py```：主函数
- ```.env```：路径配置
```shell
GPT_SOVITS_URL=http://127.0.0.1:9880/  # GPT-SoVITS接口地址
WAV2LIP_URL=http://127.0.0.1:8000/process_sync/  # Wav2Lip接口地址
VIDEO_BASE_PATH=/data/disk4/home/chenrui/chenruitmp2/video  # 模板视频路径
TEMP_VIDEO_PATH=/data/disk4/home/chenrui/chenruitmp2/temp  # 生成视频路径
GENERATED_AUDIO_PATH=/data/disk4/home/chenrui/chenruitmp2/audio  # 生成音频路径
TEMP_SRT_PATH=/data/disk4/home/chenrui/chenruitmp2/temp  # 获取生成的字幕文件路径
```
注意：这里获取字幕文件的路径需要和前面GPT-SoVITS模型生成字幕文件的路径保持一致

### 出入参
- 传入：text（用于生成音频的文本），模板视频编号（video_number）或模板视频地址（video_url）
  - 如果请求包含视频编号（video_number），则按照下面的方式访问文件（视频文件的文件名是一个数字，即编号）
  ```python
  video_path = f"{video_base_path}/{request.video_number}.mp4"
  ```
  - 如果传入的是视频文件的url（video_url），则直接从url中获取视频

- 回传：生成的同步了声音和唇部运动的视频url
- 读取文件：模板视频
- 输出文件：根据输入文本生成的音频文件、生成的唇声同步视频、生成的字幕文件（程序运行结束后会被删除，也可以手动设置不删除）
  

### API启动服务
#### 安装依赖与运行api脚本
```shell
proxychains pip install -r requirements.txt

python api.py
```

#### 发起请求
请求体模型为：
```python
class VideoRequest(BaseModel):
    text: str  # 用于生成音频的文本
    video_number: str = None  # 视频文件编号（可选）
    video_url: str = None  # 视频url（可选）
    is_srt: bool = False  # 是否生成字幕（可选）
```

接口地址-http://127.0.0.1:8010/create-video/

调用路由/create-video/，传入文本（text）与模板编号（video_number）即可返回生成视频

## 独立Stable-Diffusion文生图+视频背景替换服务
获得传入generate_custom_video的模板视频（前处理）
### 基本流程
- 用Stable-Diffusion实现文生图
- 将得到的这张图进行缩放
- 将缩放后的图片嵌入本地模板视频，获得新的模板视频

## 独立CodeFormer视频高清化服务
将generate_custom_video生成的音频和唇部运动同步的视频高清化（后处理）

### API启动流程
下载CodeFormer的权重
```shell
cd CodeFormer

proxychains python scripts/download_pretrained_models.py facelib
proxychains python scripts/download_pretrained_models.py CodeFormer
```

安装依赖
```shell
proxychains python basicsr/setup.py develop
```

运行多进程脚本
```shell
python multiprocess.py
```

运行API服务
```shell
proxychains pip install -r requirements.txt
python api.py
```

接口地址-http://127.0.0.1:8002

调用路由/process_video/,传入文本生成视频的阿里云视频链接，返回高清化的新的阿里云视频链接。