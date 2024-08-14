from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from utils.llm import get_claude_sonnet, get_custom_claude_llm
from utils.llm import get_qwen_110b, get_qwen2_72b
from utils.llm import get_gpt4
import os

task_type_2_model_name = {
    "logical_reasoning": "LOGICAL_REASONING",
    "generate_questions": "GENERATE_QUESTIONS",
    "generate_problems": "GENERATE_PROBLEMS"
}

#获取template
def load_template(task_type):
    if task_type == "logical_reasoning":
        template_path = r"./assets/logical_reasoning_prompt.txt"
    elif task_type == "generate_questions":
        template_path = r"./assets/generate_questions.txt"
    elif task_type == "generate_problems":
        template_path = r"./assets/generate_problems.txt"
    else:
        raise ValueError("task_type not found]")
    with open(template_path, "r", encoding="utf-8") as file:
        template = file.read()
    return template

#获取chain
def get_chain(task_type):
    template = load_template(task_type)
    prompt = PromptTemplate.from_template(template)
    # model_name = os.getenv("QWEN15_110B_MODEL_NAME")
    # custom_model_name = task_type_2_model_name[task_type]
    llm = get_gpt4()
    chain = prompt | llm | StrOutputParser()
    return chain