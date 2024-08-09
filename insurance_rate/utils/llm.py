import os
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_openai.embeddings.azure import AzureOpenAIEmbeddings
from utils.custom_chat_openai import CustomChatOpenAI, CustomQwenChatOpenAI, CustomChatAnthropic


TEMPERATURE = 0
MAX_TOKENS = 1200

#获取llm，带有custom的是继承重写过的，可以将llm输入输出等记录上传到sql
def get_custom_gpt4_llm(model_name, template, custom_model_name, max_retries):
    return CustomChatOpenAI(
        azure_deployment=os.getenv("OPENAI_API_DEPLOYMENT_NAME"),
        model_name=model_name,
        custom_model_name=custom_model_name,
        template=template,
        temperature=0,
        max_retries=max_retries
    )

def get_custom_qwen_llm(api_base, model_name, template, custom_model_name, max_retries):
    return CustomQwenChatOpenAI(
        openai_api_key=os.getenv('QWEN_API_KEY'),
        # openai_api_key=os.environ.get('QWEN_API_KEY')
        openai_api_base=api_base,
        custom_model_name=custom_model_name,
        template=template,
        model_name=model_name,
        temperature=0,
        max_retries=max_retries
        )


def get_custom_claude_llm(model_name, template, custom_model_name):
    return CustomChatAnthropic(
    temperature=0,
    template=template,
    model_name=model_name,
    anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY"),
    custom_model_name=custom_model_name)

def get_qwen2_72b():
    return ChatOpenAI(
        openai_api_key="empty",
        openai_api_base=os.getenv("QWEN2_72B_API_BASE"),
        model_name=os.getenv("QWEN2_72B_MODEL_NAME"),
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS
    )

def get_qwen_110b():
    return ChatOpenAI(
        openai_api_key="empty",
        openai_api_base=os.getenv("QWEN15_110B_API_BASE"),
        model_name=os.getenv("QWEN15_110B_MODEL_NAME"),
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS
        )


def get_gpt4():
    return AzureChatOpenAI(
        azure_deployment=os.getenv("OPENAI_API_DEPLOYMENT_NAME"),
        model_name=os.getenv("OPENAI_API_DEPLOYMENT_NAME"),
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS
        )


def get_claude_sonnet():
    return ChatAnthropic(
        model=os.getenv("MODEL_NAME_SONNET"),
        anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY"),
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS
        )


def get_claude_opus():
    return ChatAnthropic(
        model=os.getenv("MODEL_NAME_OPUS"),
        anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY"),
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS
        )


def get_azure_openai_embedding():
    return AzureOpenAIEmbeddings(
        azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"),
        openai_api_version=os.getenv("OPENAI_API_VERSION"),
    )


def get_azure_openai_35_llm():
    return AzureChatOpenAI(
        azure_deployment=os.getenv("OPENAI_API_DEPLOYMENT_NAME_35"),
        model_name=os.getenv("OPENAI_API_DEPLOYMENT_NAME_35"),
        temperature=0
    )