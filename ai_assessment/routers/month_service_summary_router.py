import time
from schemas.schemas import JobRequest
from utils.chain import get_chain
from utils.parse_xml import parse_xml_output, parse_xml_output_llm
from utils.produce_mq_message import produce_one_mq_message
from utils.conversation_processing import concat_conversation
from utils.custom_chat_openai import input_data

def call_record_summary_process_func(job_request:JobRequest):
    print("===========> 开始任务")
    chain = get_chain(job_request.tag)
    conversation = concat_conversation(job_request.data)
    input_data.data = conversation
    print("============> 完成数据处理，开始跑模型")
    start_time = time.time()
    res = chain.invoke({
        "content": conversation
    })
    print(f"=============>得到结果：{res}")
    end_time = time.time()
    print(f"=============>耗时：{end_time - start_time}")
    xml_dict = parse_xml_output(res, tags=["summary"], first_item_only=True)
    summary = xml_dict.get("summary", "").strip()
    if not summary:
        res, xml_dict = parse_xml_output_llm(res, tags=["summary"], first_item_only=True)
        summary = xml_dict.get("summary", "").strip()
    json_message = {
        "jobId": job_request.jobId,
        "tag": job_request.tag,
        "label_thinking": "",
        "label": "",
        "thinking": summary,
        "result": "",
        "status": "success"}
    print(f"=============> json_message: {json_message}")
    produce_one_mq_message(json_message)