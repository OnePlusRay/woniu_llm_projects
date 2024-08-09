import time
from schemas.schemas import JobRequest
from utils.chain import get_chain
from utils.parse_xml import parse_xml_output, parse_xml_output_llm, adjust_res_tag
from utils.produce_mq_message import produce_one_mq_message
from utils.conversation_processing import concat_conversation
from schemas.schemas import ParseResult, tag_2_result_mapping
from utils.custom_chat_openai import input_data

def transform_raw_conversation_2_str(raw_conversation):
    call_raw_conversations = raw_conversation["callContentList"]

    call_conversations = []
    for index, call_raw_conversation in enumerate(call_raw_conversations):
        call_conversations.append(f"第{index+1}通电话：\n" + concat_conversation(call_raw_conversation['callContent']))

    wechat_raw_conversation = raw_conversation["wechatContentList"]
    wechat_conversation = concat_conversation(wechat_raw_conversation)
    
    final_conversation = "\n\n".join(call_conversations) + "\n\n微信聊天记录：\n" + wechat_conversation
    return final_conversation


def potential_client_check_process_func(job_request:JobRequest):
    task_type = job_request.tag
    print("===========> 开始任务")
    chain = get_chain(job_request.tag)
    conversation = transform_raw_conversation_2_str(job_request.data)
    input_data.data = conversation
    print("============> 完成数据处理，开始跑模型")
    start_time = time.time()
    res = chain.invoke({
        "content": conversation
    })
    print(f"=============>得到结果：{res}")
    end_time = time.time()
    print(f"=============>耗时：{end_time - start_time}")
    xml_dict = parse_xml_output(res, tags=["thinking", "result"], first_item_only=True)
    thinking = xml_dict.get("thinking", "").strip()
    result = xml_dict.get("result", "").strip()
    if not (thinking and result):
        res, xml_dict = parse_xml_output_llm(res, tags=["thinking", "result"], first_item_only=True)
        print('------------------tag补充完整\n',res)
        thinking = xml_dict.get("thinking", "").strip()
        result = xml_dict.get("result", "").strip()
    try:
        mode = tag_2_result_mapping[task_type]
        result_test = ParseResult(parse_thinking = thinking, result_mode = mode, parse_result = result)
    except ValueError as e:
        print(e)
        mode = {"thinking":"","result":tag_2_result_mapping[task_type]}
        res = adjust_res_tag(res, tags=["thinking", "result"], result_mode=mode)
        print('------------------调整tag位置\n',res)
        xml_dict = parse_xml_output(res, tags=["thinking", "result"], first_item_only=True)
        thinking = xml_dict.get("thinking", "").strip()
        result = xml_dict.get("result", "").strip()
    json_message = {
        "jobId": job_request.jobId,
        "tag": job_request.tag,
        "thinking": thinking,
        "result": result,
        "status": "success"}
    print(f"=============> json_message: {json_message}")
    produce_one_mq_message(json_message)