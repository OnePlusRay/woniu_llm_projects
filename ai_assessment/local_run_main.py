from dotenv import load_dotenv
load_dotenv()
import os
import json
import pandas as pd
from local_run.local_run import JobRequest, call_record_quality_process_func, determine_tag
import pdb
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio
import aiohttp
import time

start_time = time.time()

def load_df_data(file_path, sheet_name):
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    data_list = []
    for row_idx, row in df.iterrows():
        data = row["data"]
        user_id = row["user_id"]
        data_list.append((data, user_id))
    return data_list

async def process_single_data(data_pair):
    data, user_id = data_pair  # data_pair是表格的一行两列拆成一个元组，其中data的结构是一个类似字典的形式（包括data，jobId，tag，其中tag决定任务类型）
    try:
        request_dict = json.loads(data)  # 解析data，使其变为python能看得懂的json字典格式

        if not isinstance(request_dict, dict):  # 如果request_dict的格式不是字典，抛出异常
            raise ValueError(f"Invalid data format for user_id {user_id}: {request_dict} (not a dict)")

        if "tag" not in request_dict or "data" not in request_dict:  # 防止得到的数据有问题
            raise ValueError(f"Invalid data format for user_id {user_id}: {request_dict}")
        
        job_request = JobRequest(  # tag出现在传入的data_pair的data中
            tag=request_dict["tag"],
            jobId="test1",
            data=request_dict["data"]
        )
        # pdb.set_trace()
        # json_message_72b = await asyncio.to_thread(call_record_quality_process_func, job_request, "72b")  # 东西扔给ai分析  # 使用 asyncio.to_thread 将同步函数转换为异步函数
        # conversation = json_message_72b["conversation"]  # 提取分析的结果1：conversation
        # judgement_basis_72b = json_message_72b["judgement_basis"]  # 提取分析的结果2：judgement_basis    
        # result_72b = json_message_72b["result"]  # 提取分析的结果3：result
        
        json_message_110b = await asyncio.to_thread(call_record_quality_process_func, job_request, "110b")
        conversation = json_message_110b["conversation"]  # 提取分析的结果1：conversation
        think_or_judge = determine_tag(job_request.tag)
        if think_or_judge == 2:
            judgement_basis_110b = json_message_110b["judgement_basis"]
        else:
            judgement_basis_110b = json_message_110b["thinking"]
        result_110b = json_message_110b["result"]

        return [user_id, conversation, judgement_basis_110b, result_110b]
    except Exception as e:
        print(f"Error processing data for user_id {user_id}: {e}")
        return None

async def process_data(data_list):
    final_res = []
    tasks = [process_single_data(data_pair) for data_pair in data_list]
    for future in asyncio.as_completed(tasks):
        result = await future
        final_res.append(result)
    return final_res

if __name__ == "__main__":
    in_file_path = "./local_run/data/POLICY_ANNUAL_入参.xlsx"
    sheet_name = "Sheet5"
    output_file_path = "./local_run/data/对比结果2.xlsx"
    data_list = load_df_data(in_file_path, sheet_name)

    final_res = asyncio.run(process_data(data_list)) 

    df = pd.DataFrame(final_res, columns=["id", "对话数据", "qwen1.5_110b_thinking", "qwen1.5_110b_result"])  # 创建一个数据框
    df.to_excel(output_file_path, index=False)  # 将数据框写到excel表格里，建成一个新的excel表


end_time = time.time()
total_time = end_time - start_time
print(f"Total running time: {total_time} seconds")