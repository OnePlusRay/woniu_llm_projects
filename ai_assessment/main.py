from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from middlewares.cors import add_cors_middleware
from schemas.schemas import JobRequest
from routers import tag_2_func_mapping
from utils.produce_mq_message import produce_one_mq_message
from utils.custom_chat_openai import input_data
app = FastAPI()
add_cors_middleware(app)


@app.post("/process/")              # 设置一个POST路由处理函数，路径为process，
async def process_inputs(job_request:JobRequest, background_tasks: BackgroundTasks):        # job_request 有三个参数：jobId: str，tag: str，data: Any
    if not job_request.data:        # 意思是如果job_request.data为空的话
        json_message = {            # 搞一个字典
        "jobId": job_request.jobId,
        "tag": job_request.tag,
        "thinking": '传入data为空值，无法得到结果',     # 得到这个结果          # =-==-=-=-=-=-=-=-=-=--==-=-=-=--=-=-=-==
        "result": '',
        "status": "false"}
        print(f"=============> json_message: {json_message}")   
        produce_one_mq_message(json_message)        # 观察用这个json_message的信息能否发出消息，包括消息ID和消息体的MD5值。
        return {
        "status": "success"
        }
    # job_request.data不为空的处理
    input_data.tag = job_request.tag
    input_data.jobId = job_request.jobId
    process_func = tag_2_func_mapping[job_request.tag]      # 将process_func 定义成 tag_2_func_mapping 中与[job_request.tag]对应的函数
    background_tasks.add_task(process_func, job_request)    # 放入process_func函数和执行该函数所需要的参数：job_request
    # 将process_func和job_request作为参数添加到后台任务中，以便在响应发送给客户端之后执行这个任务。这种方式非常适合处理那些不需要立即完成的任务，可以提高API的响应速度和用户体验
    
    return {
        "status": "success"
        }

# 自动处理异常的处理器
@app.exception_handler(HTTPException)
async def http_exception_handler(_, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"data": exc.detail},
    )
