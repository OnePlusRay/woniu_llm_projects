import os
from utils.custom_chat_openai import CustomChatOpenAI, CustomQwenChatOpenAI, CustomChatAnthropic


TEMPERATURE = 0
MAX_TOKENS = 1200


#获取llm，带有custom的是继承重写过的，可以将llm输入输出等记录上传到sql
def get_custom_gpt4(custom_model_name, max_retries):
    return CustomChatOpenAI(
        azure_deployment=os.getenv("OPENAI_API_DEPLOYMENT_NAME"),
        model_name=os.getenv("OPENAI_API_DEPLOYMENT_NAME"),
        custom_model_name=custom_model_name,
        temperature=TEMPERATURE,
        max_retries=max_retries,
        max_tokens=MAX_TOKENS
    )


def get_custom_qwen_110b(custom_model_name, template):
    return CustomQwenChatOpenAI(
        openai_api_key="empty",
        openai_api_base=os.getenv("QWEN15_110B_API_BASE"),
        custom_model_name=custom_model_name,
        template=template,
        model_name=os.getenv("QWEN15_110B_MODEL_NAME"),
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        )


def get_custom_qwen_72b(custom_model_name, template):
    return CustomQwenChatOpenAI(
        openai_api_key="empty",
        openai_api_base=os.getenv("QWEN2_72B_API_BASE"),
        custom_model_name=custom_model_name,
        template=template,
        model_name=os.getenv("QWEN2_72B_MODEL_NAME"),
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        )


def get_custom_claude(model_name, template, custom_model_name):
    return CustomChatAnthropic(
    temperature=0,
    template=template,
    model_name=model_name,
    anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY"),
    custom_model_name=custom_model_name)