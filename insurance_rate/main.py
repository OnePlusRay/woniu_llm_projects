from dotenv import load_dotenv
load_dotenv()

import os
import pdb
import requests
import tempfile
from fastapi import FastAPI, HTTPException, BackgroundTasks
from typing import Dict, Any
from schemas.schemas import RequestModel, ResponseModel
from typing import Dict, Any
from src.get_res import get_res
from urllib.parse import urlparse
# from utils.log_error_to_db import log_error_to_db, create_table_if_not_exists

app = FastAPI()

def notify_callback(callback_url: str, data: Dict[str, Any]):
    '''在程序运行结束后向callbackUrl回传消息'''
    try:
        response = requests.post(callback_url, json=data, timeout=10)  # 回传消息
        if response.status_code == 200: 
            print("Callback message sent successfully.")  # 提示回传成功
        else:
            print(f"Failed to send callback message. Status code: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"Failed to send callback: {e}")


def process_excel(request: RequestModel):
    '''处理excel表格：调主函数get_res()'''
    try:
        # 下载excel表
        response = requests.get(request.uploadUrl)
        if response.status_code != 200:  # 若状态码不是200，说明下载失败
            raise HTTPException(status_code=400, detail="Failed to download Excel file from OSS URL")
        
        # 从URL中提取原文件名
        parsed_url = urlparse(request.uploadUrl)
        original_filename = os.path.basename(parsed_url.path).replace('.xlsx', f"_{request.id}.xlsx")  # 表格初步命名为:原文件名_id.xlsx
        
        # 在临时目录中创建指定文件名的文件
        temp_dir = tempfile.mkdtemp()
        file_path = ''
        temp_file_path = os.path.join(temp_dir, original_filename)
        with open(temp_file_path, 'wb') as temp_file:
            temp_file.write(response.content)
            file_path = temp_file.name

        # 用get_res()处理excel
        output_dir_path = r"./data/real_out"
        origin_url, result_url = get_res(file_path, output_dir_path)  # 主要处理逻辑：调用主函数get_res()
        status = True
        print(f"Task for request {request.id} completed successfully!")

    except Exception as e: 
        status = False  # 若崩溃状态为False
        result_url = ''  # 若程序崩溃则downloadUrl为空
        error_message = f"Error processing file for request {request.id}: {e}"
        print(error_message)
        # create_table_if_not_exists()  # 若数据库中不存在对应的表，则创建一个新表
        # log_error_to_db(request.id, error_message)  # 将错误信息存储到数据库中

    json_message = {  # 回传的json消息
        "id": request.id,
        "downloadUrl": result_url,
        "processResult": status
    }
    print(f"=============> json_message: {json_message}")  # 打印给自己看结果

    notify_callback(request.callbackUrl, json_message)  # 程序运行结束后（无论是否成功）回传消息


@app.post("/process-excel/")
async def process_excel_endpoint(request: RequestModel, background_tasks: BackgroundTasks):
    background_tasks.add_task(process_excel, request)  # 将处理任务添加到后台任务中
    return {"id": request.id, "status": "success"}  # 返回任务ID和处理状态（无需等待）