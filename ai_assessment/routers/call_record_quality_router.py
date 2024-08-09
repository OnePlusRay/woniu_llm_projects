import time
from schemas.schemas import JobRequest
from utils.chain import get_chain, load_template
from utils.parse_xml import parse_xml_output, parse_xml_output_llm, adjust_res_tag
from utils.produce_mq_message import produce_one_mq_message
from utils.conversation_processing import concat_conversation
from schemas.schemas import ParseResult, tag_2_result_mapping
from utils.custom_chat_openai import input_data


def call_record_quality_process_func(job_request:JobRequest):
    task_type = job_request.tag
    # print("===========> 开始任务")

    chain = get_chain(job_request.tag)
    conversation = concat_conversation(job_request.data)        #该函数用于将一组对话行（lines）连接成一个字符串，每行包含角色和他们所说的话。
    input_data.data = conversation
    # print("============> 完成数据处理，开始跑模型")
    start_time = time.time()
    res = chain.invoke({
        "content": conversation
    })
    print(f"=============>得到结果：{res}")
    end_time = time.time()
    print(f"=============>耗时：{end_time - start_time}")
    xml_dict = parse_xml_output(res, tags=["judgement_basis", "result"], first_item_only=True)
    judgement_basis = xml_dict.get("judgement_basis", "").strip()
    result = xml_dict.get("result", "").strip()
    if not (judgement_basis and result):
        res, xml_dict = parse_xml_output_llm(res, tags=["judgement_basis", "result"], first_item_only=True)
        print('------------------tag补充完整\n',res)
        judgement_basis = xml_dict.get("judgement_basis", "").strip()
        result = xml_dict.get("result", "").strip()
    try:
        mode = tag_2_result_mapping[task_type]
        result_test = ParseResult(parse_thinking = judgement_basis, result_mode = mode, parse_result = result)
    except ValueError as e:
        print(e)
        mode = {"judgement_basis":"","result":tag_2_result_mapping[task_type]}
        res = adjust_res_tag(res, tags=["judgement_basis", "result"], result_mode=mode)
        print('------------------调整tag位置\n',res)
        xml_dict = parse_xml_output(res, tags=["judgement_basis", "result"], first_item_only=True)
        judgement_basis = xml_dict.get("judgement_basis", "").strip()
        result = xml_dict.get("result", "").strip()
    json_message = {
        "jobId": job_request.jobId,
        "tag": job_request.tag,
        "thinking": judgement_basis,
        "result": result,
        "status": "success"}
    print(f"=============> json_message: {json_message}")
    produce_one_mq_message(json_message)

    