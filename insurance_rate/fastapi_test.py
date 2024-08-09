from dotenv import load_dotenv
load_dotenv()

import os
import requests
from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket
from typing import Dict, Any
from schemas.schemas import RequestModel, ResponseModel
import uuid
from tempfile import NamedTemporaryFile
from typing import Dict, Any
from src.get_res import get_res
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ValidationError, validator
from enum import Enum
from typing import Any


# 定义请求体模型
class RequestModel(BaseModel):
    id: str
    uploadUrl: str
    callbackUrl: str

# 定义响应体模型
class ResponseModel(BaseModel):
    id: str
    downloadUrl: str
    processResult: bool

app = FastAPI()
# task_results: Dict[str, str] = {}
# task_status: Dict[str, str] = {}

def notify_callback(callback_url: str, data: Dict[str, Any]):
    try:
        response = requests.post(callback_url, json=data, timeout=10)  
        response.raise_for_status()
        if response.status_code == 200:
            print("Callback message sent successfully.")
        else:
            print(f"Failed to send callback message. Status code: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"Failed to send callback: {e}")

def process_excel(request: RequestModel):
    '''处理excel表格：调主函数get_res()'''
    
    try:
        request_id = request.id
        file_url = request.uploadUrl
        # task_status[request_id] = "Processing"
        # 下载excel表
        response = requests.get(file_url)
        if response.status_code != 200:  # 若状态码不是200，说明下载失败
            raise HTTPException(status_code=400, detail="Failed to download Excel file from OSS URL")
        
        file_path = ''
        with NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp:
            tmp.write(response.content)
            file_path = tmp.name

        # 用get_res()处理excel
        output_dir_path = r"./data/real_out"
        test = False  # 是否为小样本测试
        is_debug = False
        origin_url, result_url = get_res(file_path, output_dir_path, test, is_debug)
        status = True
        print(f"Task for request {request.id} completed with result: {result_url}")

    except Exception as e:
        status = False
        result_url = ''
        print(f"Error processing file for request {request.id}: {e}")

    json_message = {
        "id": request.id,
        "downloadUrl": result_url,
        "processResult": status
    }
    print(f"=============> json_message: {json_message}")

    notify_callback(request.callbackUrl, json_message)

    # return result_url



@app.post("/process-excel/")
async def process_excel_endpoint(request: RequestModel, background_tasks: BackgroundTasks):
    
    # 将处理任务添加到后台任务中
    background_tasks.add_task(process_excel, request)

    # background_tasks.add_task(task_callback, request.id)

    # task_status[request.id] = "Pending"
        
    # 返回任务ID
    return {"id": request.id, "status": "success"}
    
    # 不用后台处理，直接运行
    # result_url = process_excel(request.uploadUrl)
  
# @app.get("/task-status/{request_id}", response_model=ResponseModel)
# async def get_task_status(request_id: str):
#     if request_id in task_status:
#         status = task_status[request_id]
#         if status == "Completed":
#             return {"id": request_id, "downloadUrl": task_results[request_id]}
#         elif status == "Failed":
#             return {"id": request_id, "downloadUrl": task_results[request_id]}
#         else:
#             return {"id": request_id, "downloadUrl": status}
#     else:
#         raise HTTPException(status_code=404, detail="Task not found")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="10.0.0.25", port=8080)