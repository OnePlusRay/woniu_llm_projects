from fastapi import FastAPI
from pydantic import BaseModel
import httpx
from httpx import HTTPStatusError, RequestError
from fastapi.responses import JSONResponse
import os
import uuid
import logging
from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv()

# 使用环境变量
gpt_sovits_url = os.getenv("GPT_SOVITS_URL")
wav2lip_url = os.getenv("WAV2LIP_URL")
video_base_path = os.getenv("VIDEO_BASE_PATH")
temp_video_path = os.getenv("TEMP_VIDEO_PATH")
generated_audio_path = os.getenv("GENERATED_AUDIO_PATH")
temp_srt_path = os.getenv("TEMP_SRT_PATH")


# 检查并创建文件夹
if not os.path.exists(temp_video_path):
    os.makedirs(temp_video_path)

if not os.path.exists(generated_audio_path):
    os.makedirs(generated_audio_path)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

class VideoRequest(BaseModel):
    text: str
    video_number: str = None 
    video_url: str = None
    is_srt: bool = False

if not os.path.exists(video_base_path):
    os.makedirs(video_base_path, exist_ok=True)

@app.post("/create-video/")
async def create_synced_video(request: VideoRequest):
    unique_id = str(uuid.uuid4())  # 生成一个唯一ID
    temp_audio_path = f'{generated_audio_path}/generated_audio_{unique_id}.wav'
    logger.info("Received request to create video for text: %s", request.text)

    try:
        async with httpx.AsyncClient(timeout=1200.0) as client:
            if request.is_srt:
                params = {"text": request.text, "text_language": "zh", "cut_punc": "。", "unique_id": unique_id}
                audio_response = await client.post(gpt_sovits_url+"get_srt", json=params)
            else:
                params = {"text": request.text, "text_language": "zh", "cut_punc": "。"}
                audio_response = await client.post(gpt_sovits_url, json=params)
            audio_response.raise_for_status()

            logger.info("Successfully generated audio.")
            with open(temp_audio_path, 'wb') as f:
                f.write(audio_response.content)

            if request.video_number:
                video_path = f"{video_base_path}/{request.video_number}.mp4"
            else:
                temp_video_file = f'{temp_video_path}/downloaded_video_{unique_id}.mp4'
                video_response = await client.get(request.video_url)
                with open(temp_video_file, 'wb') as f:
                    f.write(video_response.content)
                video_path = temp_video_file

            if request.is_srt:
                sync_request = {"audio_path": temp_audio_path, "video_path": video_path, "srt_path": f"{temp_srt_path}/{unique_id}.srt"}              
            else:
                sync_request = {"audio_path": temp_audio_path, "video_path": video_path}
                
            video_response = await client.post(wav2lip_url, json=sync_request)
            video_response.raise_for_status()

            logger.info("Successfully processed video creation.")
            return {"file_url": video_response.json()['file_url']}
        
    except HTTPStatusError as http_err:
        return JSONResponse(status_code=http_err.response.status_code, content={"message": str(http_err)})
    except RequestError as req_err:
        return JSONResponse(status_code=500, content={"message": f"Network error: {str(req_err)}"})
    except IOError as io_err:
        return JSONResponse(status_code=500, content={"message": f"File I/O error: {str(io_err)}"})
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": f"An error occurred: {str(e)}"})
    finally:
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
        if 'temp_video_file' in locals() and os.path.exists(temp_video_file):
            os.remove(temp_video_file)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)