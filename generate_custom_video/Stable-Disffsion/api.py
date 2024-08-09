from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import subprocess
import base64
import uuid
from dotenv import load_dotenv
import oss2
import os
import httpx
from httpx import HTTPStatusError, RequestError
from fastapi.responses import JSONResponse
from fastapi.concurrency import run_in_threadpool

# 加载 .env 文件
load_dotenv()

output_path = os.getenv("OUTPUT_PATH")
template_path = os.getenv("TEMPLATE_PATH")

# 检查并创建文件夹
if not os.path.exists(output_path):
    os.makedirs(output_path)

if not os.path.exists(template_path):
    os.makedirs(template_path)

# 阿里云OSS配置
auth = oss2.Auth(os.getenv('ACCESS_KEY_ID'), os.getenv('ACCESS_KEY_SECRET'))
bucket = oss2.Bucket(auth, os.getenv('ENDPOINT'), os.getenv('OSS_BUCKET_NAME'))

app = FastAPI()

class VideoRequest(BaseModel):
    prompt: str

async def ff_gen_template_video(clip_bg_fp, final_template_video_fp, resized_image_path, video_template_path):
    subprocess.run(['ffmpeg', '-i', clip_bg_fp, '-vf', 'scale=1080:1920', '-y', resized_image_path])
    subprocess.run(['ffmpeg', '-y', '-loop', '1', '-i', resized_image_path, '-i', video_template_path, '-filter_complex', '[0:v][1:v]overlay=shortest=1', final_template_video_fp])
    subprocess.run(['rm', resized_image_path])

async def sd_gen_template_video(prompt, final_template_video_fp, png_output_path, resized_image_path, video_template_path):
    url = "http://sd.insnail.com"
    async with httpx.AsyncClient() as client:
        response = await client.get(f'{url}/sdapi/v1/options')
        response.raise_for_status()

        if "realisticStockPhoto" not in response.json()['sd_model_checkpoint']:
            option_payload = {
                "sd_model_checkpoint": "realisticStockPhoto_v30SD15",
            }
            option_response = await client.post(url=f'{url}/sdapi/v1/options', json=option_payload)
            print(option_response.json())

        payload = {
            "prompt": prompt,
            "steps": 20,
            "width": 720,
            "height": 1280,
            "refiner_checkpoint": "realisticStockPhoto_v30SD15",
            "refiner_switch_at": 0.8
        }
        response = await client.post(f'{url}/sdapi/v1/txt2img', json=payload)
        response.raise_for_status()
        with open(png_output_path, 'wb') as f:
            f.write(base64.b64decode(response.json()['images'][0]))

        await ff_gen_template_video(png_output_path, final_template_video_fp, resized_image_path, video_template_path)

async def upload_mp4_to_oss(file_path: str):
    if not file_path.endswith('.mp4'):
        raise ValueError("The file is not an MP4 file.")

    id = uuid.uuid1()
    save_path = f"generate_custom_video/results/{str(id)}/result_voice.mp4"
    file_url = f"https://{os.getenv('OSS_BUCKET_NAME')}.{os.getenv('ENDPOINT')}/generate_custom_video/results/{str(id)}/result_voice.mp4"

    try:
        # 由于oss2不支持异步，我们在线程池中运行同步代码
        await run_in_threadpool(upload_file, file_path, save_path)
        return file_url
    except oss2.exceptions.OssError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except IOError as e:
        raise HTTPException(status_code=500, detail=f"Error opening file: {e}")

def upload_file(file_path, save_path):
    with open(file_path, 'rb') as file_data:
        bucket.put_object(save_path, file_data)

@app.post("/generate_video/")
async def generate_video(request: VideoRequest):
    try:
        unique_id = str(uuid.uuid4())
        resized_image_path = f"{template_path}/{unique_id}_resized_image.png"
        video_template_path = f"{template_path}/video_template_with_mask.mov"
        png_output_path = f"{output_path}/{unique_id}_output.png"
        video_output_path = f"{output_path}/{unique_id}_ff_template_gen.mp4"
        prompt_with_tags = f"Anchemix house,Anchemix realistic,no humans,scenery,indoors,realistic,shoot by nikon z9,{request.prompt},shadow,<lora:Anchemix_realistic_house_SDXL_V1-000010:0.7>"
        
        await sd_gen_template_video(prompt_with_tags, video_output_path, png_output_path, resized_image_path, video_template_path)
        file_url = await upload_mp4_to_oss(video_output_path)
        return {"message": "Video generation initiated", "file_url": file_url}
    except HTTPStatusError as http_err:
        return JSONResponse(status_code=http_err.response.status_code, content={"message": str(http_err)})
    except RequestError as req_err:
        return JSONResponse(status_code=500, content={"message": f"Network error: {str(req_err)}"})
    except IOError as io_err:
        return JSONResponse(status_code=500, content={"message": f"File I/O error: {str(io_err)}"})
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": f"An error occurred: {str(e)}"})
    finally:
        if os.path.exists(resized_image_path):
            os.remove(resized_image_path)
        if os.path.exists(png_output_path):
            os.remove(png_output_path)
        if os.path.exists(video_output_path):
            os.remove(video_output_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)