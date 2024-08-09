from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from utils.llm import get_qwen_110b, get_qwen_72b
from utils.llm_with_sql import get_custom_qwen_110b, get_custom_qwen_72b


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
    "NEWEST_CALL_RECORD": r"./assets/newest_call_record_template.txt",
    "DAILY_SERVICE_SUMMARY": r"./assets/daily_service_summary_prompt.txt",
    "MONTH_SERVICE_SUMMARY": r"./assets/month_service_summary_prompt.txt"
}


def load_template(tag:str):
    template_path = tag_2_path_mapping[tag]
    with open(template_path, "r", encoding="utf-8") as file:
        template = file.read()
    return template


def get_chain(tag:str):
    prompt_template = load_template(tag)
    prompt = PromptTemplate.from_template(prompt_template)
    # llm = get_qwen_72b()
    llm = get_custom_qwen_110b(tag, template = prompt_template)
    chain = prompt | llm | StrOutputParser()
    return chain