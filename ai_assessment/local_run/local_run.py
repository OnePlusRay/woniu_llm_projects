import re
import time
from schemas.schemas import JobRequest
from utils.conversation_processing import concat_conversation
from schemas.schemas import ParseResult, tag_2_result_mapping
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from utils.llm import get_qwen_110b, get_qwen_72b


tag_2_path_mapping = {
    "ANNUAL_REPORT_QUALITY": r"./assets/annual_report_quality_prompt.txt",
    "POTENTIAL_CHECK": r"./assets/potential_client_check_prompt.txt",
    "SUMMARY": r"./assets/call_record_summary_prompt.txt",
    "INITIAL_CALL": r"./assets/initial_call_quality_prompt.txt",
    "POLICY_ANNUAL": r"./assets/call_record_quality_prompt.txt",
    "FULLFILL_RESULT": r"./assets/fulfill_result.txt",
    "ADJUST_RESULT_TAG": r"./assets/adjust_result_tag.txt",
    "ORIGINAL_CALL_RECORD": r"./assets/original_call_record_prompt.txt",
    "HUBIN_CALL_RECORD": r"./assets/original_hubin_prompt.txt",
    "NEWEST_CALL_RECORD": r"./assets/newest_call_record_template.txt"
}

def determine_tag(tag: str):
    template = load_template(tag)

    if "<thinking>" in template and "</thinking>" in template:
        return 1
    elif "<judgement_basis>" in template and "</judgement_basis>" in template:
        return 2
    else:
        return 0

def load_template(tag:str):
    template_path = tag_2_path_mapping[tag]
    with open(template_path, "r", encoding="utf-8") as file:
        template = file.read()
    return template


def get_chain(tag:str, model_name):
    prompt_template = load_template(tag)
    prompt = PromptTemplate.from_template(prompt_template)

    if model_name == "72b":
        llm = get_qwen_72b()
    else:
        llm = get_qwen_110b()
    chain = prompt | llm | StrOutputParser()
    return chain


def parse_xml_output(output_str:str, tags:list, first_item_only=False):
    xml_dict = {}
    for tag in tags:
        texts = re.findall(rf'<{tag}>(.*?)</{tag}>', output_str, re.DOTALL)
        if texts:
            xml_dict[tag] = texts[0] if first_item_only else texts
    return xml_dict


def parse_xml_output_llm(output_str:str, tags:list, first_item_only=False):
    xml_dict = {}
    chain = get_chain("FULLFILL_RESULT", "110b")
    res = chain.invoke({'xml_contents': output_str, 'specific_tags': tags})
    xml_dict = parse_xml_output(res,tags,first_item_only)
    return res, xml_dict


def adjust_res_tag(output_str: str, tags:list, result_mode):
    chain = get_chain("ADJUST_RESULT_TAG", "110b")
    res = chain.invoke({'xml_contents': output_str, 'specific_tags': tags, "required_mode": result_mode})
    return res


def call_record_quality_process_func(job_request:JobRequest, model_name):
    task_type = job_request.tag
    # print("===========> 开始任务")
    # 判断txt里是thinking还是judgement_basis：1为thinking，2为judgement_basis
    think_or_judge = determine_tag(job_request.tag)
    if think_or_judge == 2:
        chain = get_chain(job_request.tag, model_name)
        conversation = concat_conversation(job_request.data)
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
            "conversation": conversation,
            "tag": job_request.tag,
            "judgement_basis": judgement_basis,
            "result": result,
            "status": "success"}
        print(f"=============> json_message: {json_message}")
        return json_message
    else: 
        chain = get_chain(job_request.tag, model_name)
        conversation = concat_conversation(job_request.data)
        # print("============> 完成数据处理，开始跑模型")
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
            "conversation": conversation,
            "tag": job_request.tag,
            "thinking": thinking,
            "result": result,
            "status": "success"}
        print(f"=============> json_message: {json_message}")
        return json_message
    