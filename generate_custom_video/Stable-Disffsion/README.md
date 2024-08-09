# 项目API启动流程文档

## 目录
- [项目API启动流程文档](#项目api启动流程文档)
  - [目录](#目录)
  - [简介](#简介)
    - [基本流程](#基本流程)
    - [出入参](#出入参)
    - [文档结构](#文档结构)
    - [辅助链接](#辅助链接)
  - [配置安装](#配置安装)
    - [克隆仓库](#克隆仓库)
    - [软件依赖](#软件依赖)
  - [本地启动项目](#本地启动项目)
    - [注意](#注意)

## 简介
### 基本流程
- 用Stable-Diffusion实现文生图
- 将得到的这张图进行缩放
- 将缩放后的图片嵌入本地模板视频，获得新的模板视频

### 出入参
- 传入：prompt
- 回传：嵌入图片背景后的模板视频url
- 读取文件：模板视频
- 输出文件：sd生成原图、缩放后的图片、嵌入背景的模板视频（这些都是中间变量）
  

### 文档结构
- content：储存模板视频和缩放后的图片（缩放后的图片是中间变量，最终会删除）
- temp：储存SD生成的图片、最终输出的更换了背景的新模板视频
- images：储存readme文档中需要插入的照片
- requirements.txt：项目依赖库
- .env：储存环境变量，主要包括阿里云oss_bucket、文件路径配置

### 辅助链接
SD地址：http://sd.insnail.com
SD的API接口文档：http://sd.insnail.com/docs#/

## 配置安装
### 克隆仓库
```shell
git clone -b develop https://git.woniubaoxian.com/ai-project/generate_custom_video.git
```

### 软件依赖
- Python依赖库：requirements.txt（使用国内清华源安装更快）
```shell
pip install -r requirements.txt
```

## 本地启动项目
- 项目启动：直接运行api脚本
```shell
python api.py
```

- 用Apifox在```http://0.0.0.0:8005/generate_video/```发起请求，请求体模型为
```python
class VideoRequest(BaseModel):
    prompt: str
```
即只需输入一段提示词（也可以不输入，那就是默认使用提示词）。
具体的请求发起操作见下图：
![Alt text](images/start.jpg)

### 注意
- 若出现```500 Internal Server Error```的报错，增加超时时间设置也许可以解决，默认的超时时间是5秒，建议增加至10秒以上（SD的文生图时间在5-10秒），具体的代码位置见下图：
![Alt text](images/param.jpg)




