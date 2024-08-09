import os
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain_anthropic import ChatAnthropic


TEMPERATURE = 0
MAX_TOKENS = 1200


def get_qwen_110b():
    return ChatOpenAI(
        openai_api_key="empty",
        openai_api_base=os.getenv("QWEN15_110B_API_BASE"),
        model_name=os.getenv("QWEN15_110B_MODEL_NAME"),
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS
        )


def get_qwen_72b():
    return ChatOpenAI(
        openai_api_key="empty",
        openai_api_base=os.getenv("QWEN2_72B_API_BASE"),
        model_name=os.getenv("QWEN2_72B_MODEL_NAME"),
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