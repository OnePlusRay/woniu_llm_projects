import os
import requests
from fastapi import FastAPI, HTTPException
from tempfile import NamedTemporaryFile
from typing import Dict, Any
from src.get_res import get_res

# def download_excel(url: str) -> str:
#     '''步骤一：下载uploadUrl上的表格'''
#     response = requests.get(url)
#     if response.status_code != 200:  # 若状态码不是200，说明下载失败
#         raise HTTPException(status_code=400, detail="Failed to download Excel file from OSS URL")
    
#     with NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp:
#         tmp.write(response.content)
#         return tmp.name  # tmp.name：临时文件的file_path

# 放main了，这个.py没用到

def process_excel(request_id: str, file_url: str) -> str:
    '''处理excel表格：调主函数get_res()'''
    try:
        response = requests.get(file_url)
        if response.status_code != 200:  # 若状态码不是200，说明下载失败
            raise HTTPException(status_code=400, detail="Failed to download Excel file from OSS URL")
        
        file_path = ''
        with NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp:
            tmp.write(response.content)
            file_path = tmp.name

        output_dir_path = r"./data/real_out"
        test = False  # 是否为小样本测试
        origin_url, result_url = get_res(file_path, output_dir_path, test)
        print(f"Task for request {request_id} completed with result: {result_url}")
    except Exception as e:
        print(f"Error processing file for request {request_id}: {e}")
    return result_url
    

# def upload_to_oss(file_path: str) -> str:
#     '''步骤三：把处理好的excel表格上传到oss并返回downloadUrl'''
#     # 示例：将文件上传到 OSS 并返回新的 OSS URL
#     # 你需要根据实际情况实现上传逻辑
#     # 这里假设上传成功并返回一个示例 URL
#     return "https://insnail-asr-nls.oss-cn-shenzhen.aliyuncs.com/flatten_table/data/result/国富人寿小红花致夏版_flatten_1718859725.xlsx.xlsx"
