from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from utils.llm import get_claude_sonnet, get_custom_claude_llm
from utils.llm import get_qwen_110b, get_qwen2_72b
import os

task_type_2_model_name = {
    "table_structure_recognize": "TABLE_STRUCTURE_RECOGNIZE",
    "table_structure_recognize_no_main_title": "TABLE_STRUCTURE_RECOGNIZE_NO_MAIN_TITLE",
    "match_key": "MATCH_KEY",
    "match_key_multi_class": "MATCH_KEY_MULTI_CLASS",
    "get_enum_from_maintitle": "GET_ENUM_FROM_MAINTITLE",
    "match_factor_enum": "MATCH_FACTOR_ENUM"
}

#获取template
def load_template(task_type):
    if task_type == "table_structure_recognize":
        template_path = r"./assets/table_structure_recognize_prompt.txt"
    elif task_type == "table_structure_recognize_no_main_title":
        template_path = r"./assets/table_structure_recognize_prompt_no_main_title.txt"
    elif task_type == "match_key":
        template_path = r"./assets/match_key_prompt.txt"
    elif task_type == "match_key_multi_class":
        template_path = r"./assets/match_key_prompt_multi_class.txt"
    elif task_type == "get_enum_from_maintitle":
        template_path = r"./assets/get_enum_from_maintitle.txt"
    elif task_type == "match_factor_enum":
        template_path = r"./assets/match_factor_enum_prompt.txt"
    else:
        raise ValueError("task_type must be in [table_structure_recognize, table_structure_recognize_no_main_title, match_key, match_key_multi_class, match_factor_enum]")
    with open(template_path, "r", encoding="utf-8") as file:
        template = file.read()
    return template

#获取chain
def get_chain(task_type):
    template = load_template(task_type)
    prompt = PromptTemplate.from_template(template)
    # model_name = os.getenv("QWEN15_110B_MODEL_NAME")
    # custom_model_name = task_type_2_model_name[task_type]
    llm = get_qwen2_72b()
    chain = prompt | llm | StrOutputParser()
    return chain