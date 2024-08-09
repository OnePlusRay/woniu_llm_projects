import os
import json
import time
import requests
import tiktoken
from datetime import datetime
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.outputs import ChatGeneration,ChatGenerationChunk,LLMResult,RunInfo
from langchain_core.outputs.chat_generation import ChatGeneration
from langchain_core.language_models.base import LanguageModelInput
from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.base import Runnable
from langchain_core.runnables.config import ensure_config
from langchain_core.runnables.utils import Input, Output
from langchain_core.prompt_values import ChatPromptValue, PromptValue, StringPromptValue
from langchain_core.callbacks import CallbackManager,Callbacks
from langchain_core.load import dumpd
from sqlalchemy import create_engine, Column, Integer, String, Text, Float, Time, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from typing import Any,Dict,Iterator,List,Optional,cast
import pymysql
pymysql.install_as_MySQLdb()
# 定义模型的基类
Base = declarative_base()


class InputData():
    data: str
    jobId: str
    tag: str
input_data = InputData()


class Prompt(Base):
    __tablename__ = 'ai_assessment_all_model_prompt'
    id = Column(Integer, primary_key=True, autoincrement=True)
    data = Column(Text)
    jobId = Column(Text)
    tag = Column(Text)
    custom_model_name = Column(Text)
    template = Column(Text)
    model_input = Column(Text)
    model_output = Column(Text)
    start_time = Column(DateTime)
    end_time = Column(DateTime)
    model_name = Column(Text)
    model_input_token = Column(Integer)
    model_output_token= Column(Integer)

class PromptSQLService():
    def __init__(self):
        self.engine = self._create_table_if_not_exists()
        self.Session = sessionmaker(bind=self.engine)

    def _create_table_if_not_exists(self):
        try:
            DATABASE_URI = f"""mysql+mysqlconnector://{os.getenv('MYSQL_USERNAME')}:{os.getenv('MYSQL_PASSWD')}@{os.getenv('MYSQL_HOST')}:{os.getenv('MYSQL_PORT')}/{os.getenv('MYSQL_DATABASE')}"""
            engine = create_engine(DATABASE_URI)
            Base.metadata.create_all(engine)
            return engine
        except Exception as e:
            # 添加异常处理
            print(f"Error connecting to the database: {e}")
            raise

    def _insert_cost_data(self, cost_data):
        post_url = 'https://qasweb.insnail.cn/front/api/logging/singlePiece'
        data = {
            "data": cost_data,
            "isInsert": 1,
            "table": "front_ai_chain_data_test"
        }
        json_data = json.dumps(data)
        headers = {'Content-Type': 'application/json'}
        response = requests.post(post_url, data=json_data, headers=headers)
        if response.status_code != 200:
            print("AI cost post error", response.status_code)
            print("Response:", response.text)

    def insert_data(self, data):
        session = self.Session()
        try:
            new_entry = Prompt(
                data = input_data.data,
                jobId = input_data.jobId,
                tag = input_data.tag,
                custom_model_name = data.get("custom_model_name", ""),
                template=data.get("template", ""), 
                model_input=data.get("model_input", ""), 
                model_output=data.get("model_output", ""),
                start_time = data.get("start_time", ""),
                end_time = data.get("end_time", ""),
                model_name = data.get("model_name", ""),
                model_input_token = data.get("model_input_token", ""),
                model_output_token = data.get("model_output_token", "")
                )
            session.add(new_entry)
            session.commit()
        finally:
            session.close()

    def get_title(self, title: str):
        session = self.Session()
        try:
            results = session.query(Prompt).filter(Prompt.question.in_(title)).all()
            return results
        finally:
            session.close()

    def get_all_prompt(self):
        session = self.Session()
        try:
            results = session.query(Prompt).all()
            if len(results) == 0:
                return []
            else:
                return results
        finally:
            session.close()

prompt_sql_service = PromptSQLService()


MODEL_COST_PER_1K_TOKENS = {
    # GPT-4 input
    "gpt-4": 0.03,
    "gpt-4-32k": 0.06,
    "gpt-4-1106-preview": 0.01,
    # GPT-4 output
    "gpt-4-completion": 0.06,
    "gpt-4-32k-completion": 0.12,
    "gpt-4-1106-preview-completion": 0.03,
    # GPT-3.5 input
    "gpt-35-turbo": 0.0015,
    # GPT-3.5 output
    "gpt-35-turbo-completion": 0.002
}


class CustomChatOpenAI(AzureChatOpenAI, Runnable):
    custom_model_name: str
    template: str

    def _get_cur_time(self):
        current_time = datetime.now()
        return current_time

    def _get_offical_model_name(self):
        if self.model_name == "gpt4-1106-preview":
            return "gpt-4-1106-preview"
        return self.model_name

    def special_save_train_data(self, model_input: str, model_output: str, start_time, end_time):
        encoding = tiktoken.encoding_for_model(self._get_offical_model_name())
        model_input_token = len(encoding.encode(model_input))
        model_output_token = len(encoding.encode(model_output))
        data = {
            "custom_model_name": self.custom_model_name,
            "template": self.template,
            "model_input": model_input,
            "model_output": model_output,
            "start_time": start_time,
            "end_time": end_time,
            "model_name": self.model_name, 
            "model_input_token": model_input_token,
            "model_output_token": model_output_token
            }
        prompt_sql_service.insert_data(data)

    def _get_text(self, input_object):
        if isinstance(input_object, StringPromptValue):
            return input_object.text
        if isinstance(input_object, ChatPromptValue):
            return input_object.messages[0].content
        if isinstance(input_object, ChatGeneration):
            return input_object.text

    def invoke(
        self,
        input: LanguageModelInput,
        config: Optional[RunnableConfig] = None,
        *,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> BaseMessage:
        config = ensure_config(config)
        start_time = self._get_cur_time()
        response = self.generate_prompt(
                [self._convert_input(input)],
                stop=stop,
                callbacks=config.get("callbacks"),
                tags=config.get("tags"),
                metadata=config.get("metadata"),
                run_name=config.get("run_name"),
                **kwargs,
            ).generations[0][0]
        end_time = self._get_cur_time()
        input_text = self._get_text(input)
        output_text = self._get_text(response)
        self.special_save_train_data(input_text, output_text, start_time, end_time)
        return cast(
            ChatGeneration,
            response,
        ).message
    
    def stream(
        self,
        input: Input,
        config: Optional[RunnableConfig] = None,
        **kwargs: Optional[Any],
    ) -> Iterator[Output]:
        """
        Default implementation of stream, which calls invoke.
        Subclasses should override this method if they support streaming output.
        """
        yield self.invoke(input, config, **kwargs)


class CustomQwenChatOpenAI(ChatOpenAI, Runnable):
    custom_model_name: str
    template: str

    def _get_cur_time(self):
        current_time = datetime.now()
        return current_time

    def special_save_train_data(self, model_input: str, model_output: str, start_time, end_time):
        data = {
            "custom_model_name": self.custom_model_name,
            "template": self.template,
            "model_input": model_input,
            "model_output": model_output,
            "start_time": start_time,
            "end_time": end_time,
            "model_name": self.model_name, 
            "model_input_token": 0,
            "model_output_token": 0
            }
        prompt_sql_service.insert_data(data)

    def _get_text(self, input_object):
        if isinstance(input_object, StringPromptValue):
            return input_object.text
        if isinstance(input_object, ChatPromptValue):
            return input_object.messages[0].content
        if isinstance(input_object, ChatGeneration):
            return input_object.text

    def invoke(
        self,
        input: LanguageModelInput,
        config: Optional[RunnableConfig] = None,
        *,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> BaseMessage:
        config = ensure_config(config)
        start_time = self._get_cur_time()
        response = self.generate_prompt(
                [self._convert_input(input)],
                stop=stop,
                callbacks=config.get("callbacks"),
                tags=config.get("tags"),
                metadata=config.get("metadata"),
                run_name=config.get("run_name"),
                **kwargs,
            ).generations[0][0]
        end_time = self._get_cur_time()
        input_text = self._get_text(input)
        output_text = self._get_text(response)
        self.special_save_train_data(input_text, output_text, start_time, end_time)
        return cast(
            ChatGeneration,
            response,
        ).message
    
    def stream(
        self,
        input: Input,
        config: Optional[RunnableConfig] = None,
        **kwargs: Optional[Any],
    ) -> Iterator[Output]:
        """
        Default implementation of stream, which calls invoke.
        Subclasses should override this method if they support streaming output.
        """
        yield self.invoke(input, config, **kwargs)


class CustomChatAnthropic(ChatAnthropic):
    custom_model_name: str
    template: str

    def _get_cur_time(self):
        current_time = datetime.now()
        return current_time

    def special_save_train_data(self, model_input: str, model_output: str, start_time, end_time,input_tokens,output_tokens):
        data = {
            "custom_model_name": self.custom_model_name,
            "template": self.template,
            "model_input": model_input,
            "model_output": model_output,
            "start_time": start_time,
            "end_time": end_time,
            "model_name": self.model, 
            "model_input_token": input_tokens,
            "model_output_token": output_tokens
            }
        prompt_sql_service.insert_data(data)

    def _get_text(self, input_object):
        if isinstance(input_object, StringPromptValue):
            return input_object.text
        if isinstance(input_object, ChatPromptValue):
            return input_object.messages[0].content
        if isinstance(input_object, ChatGeneration):
            return input_object.text

    def _combine_llm_outputs(self,llm_outputs_list):
        input_t=0
        output_t=0
        for output in llm_outputs_list:
            input_t+=output['usage'].input_tokens
            output_t+=output['usage'].output_tokens
        return {'input_tokens':input_t,'output_tokens':output_t}

    def generate(
        self,
        messages: List[List[BaseMessage]],
        stop: Optional[List[str]] = None,
        callbacks: Callbacks = None,
        *,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        run_name: Optional[str] = None,
        **kwargs: Any,
    ) -> LLMResult:
        """Pass a sequence of prompts to the model and return model generations.

        This method should make use of batched calls for models that expose a batched
        API.

        Use this method when you want to:
            1. take advantage of batched calls,
            2. need more output from the model than just the top generated value,
            3. are building chains that are agnostic to the underlying language model
                type (e.g., pure text completion models vs chat models).

        Args:
            messages: List of list of messages.
            stop: Stop words to use when generating. Model output is cut off at the
                first occurrence of any of these substrings.
            callbacks: Callbacks to pass through. Used for executing additional
                functionality, such as logging or streaming, throughout generation.
            **kwargs: Arbitrary additional keyword arguments. These are usually passed
                to the model provider API call.

        Returns:
            An LLMResult, which contains a list of candidate Generations for each input
                prompt and additional model provider-specific output.
        """
        params = self._get_invocation_params(stop=stop, **kwargs)
        options = {"stop": stop}

        callback_manager = CallbackManager.configure(
            callbacks,
            self.callbacks,
            self.verbose,
            tags,
            self.tags,
            metadata,
            self.metadata,
        )
        run_managers = callback_manager.on_chat_model_start(
            dumpd(self),
            messages,
            invocation_params=params,
            options=options,
            name=run_name,
            batch_size=len(messages),
        )
        results = []
        for i, m in enumerate(messages):
            try:
                results.append(
                    self._generate_with_cache(
                        m,
                        stop=stop,
                        run_manager=run_managers[i] if run_managers else None,
                        **kwargs,
                    )
                )
            except BaseException as e:
                if run_managers:
                    run_managers[i].on_llm_error(e, response=LLMResult(generations=[]))
                raise e
        flattened_outputs = [
            LLMResult(generations=[res.generations], llm_output=res.llm_output)
            for res in results
        ]
        llm_output = self._combine_llm_outputs([res.llm_output for res in results])
        generations = [res.generations for res in results]
        output = LLMResult(generations=generations, llm_output=llm_output)
        if run_managers:
            run_infos = []
            for manager, flattened_output in zip(run_managers, flattened_outputs):
                manager.on_llm_end(flattened_output)
                run_infos.append(RunInfo(run_id=manager.run_id))
            output.run = run_infos
        return output

    def invoke(
        self,
        input: LanguageModelInput,
        config: Optional[RunnableConfig] = None,
        *,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> BaseMessage:
        config = ensure_config(config)
        start_time = self._get_cur_time()
        result=self.generate_prompt(
                [self._convert_input(input)],
                stop=stop,
                callbacks=config.get("callbacks"),
                tags=config.get("tags"),
                metadata=config.get("metadata"),
                run_name=config.get("run_name"),
                **kwargs,
            )
        response = result.generations[0][0]
        input_tokens=result.llm_output['input_tokens']
        output_tokens=result.llm_output['output_tokens']
        end_time = self._get_cur_time()
        input_text = self._get_text(input)
        output_text = self._get_text(response)
        #print('input_tokens:',input_tokens,'output_tokensut',output_tokens,'\n')
        self.special_save_train_data(input_text, output_text, start_time, end_time,input_tokens,output_tokens)
        return cast(
            ChatGeneration,
            response,
        ).message
    
    def stream(
        self,
        input: Input,
        config: Optional[RunnableConfig] = None,
        **kwargs: Optional[Any],
    ) -> Iterator[Output]:
        """
        Default implementation of stream, which calls invoke.
        Subclasses should override this method if they support streaming output.
        """
        yield self.invoke(input, config, **kwargs)

    def count_tokens(self,input:Input):
        return self._client.count_tokens(input)
    
    
    

    
