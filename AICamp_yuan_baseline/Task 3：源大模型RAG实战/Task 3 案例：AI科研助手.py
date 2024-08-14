# 导入所需的库
import torch
import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.text_splitter import RecursiveCharacterTextSplitter

from typing import Any, List, Optional


# 向量模型下载
from modelscope import snapshot_download
model_dir = snapshot_download('AI-ModelScope/bge-small-en-v1.5', cache_dir='./')

# 源大模型下载
from modelscope import snapshot_download
model_dir = snapshot_download('IEITYuan/Yuan2-2B-Mars-hf', cache_dir='./')
# model_dir = snapshot_download('IEITYuan/Yuan2-2B-July-hf', cache_dir='./')

# 定义模型路径
model_path = './IEITYuan/Yuan2-2B-Mars-hf'
# path = './IEITYuan/Yuan2-2B-July-hf'

# 定义向量模型路径
embedding_model_path = './AI-ModelScope/bge-small-en-v1___5'

# 定义模型数据类型
torch_dtype = torch.bfloat16 # A10
# torch_dtype = torch.float16 # P100


# 定义源大模型类
class Yuan2_LLM(LLM):
    """
    class for Yuan2_LLM
    """
    tokenizer: AutoTokenizer = None
    model: AutoModelForCausalLM = None

    def __init__(self, mode_path :str):
        super().__init__()

        # 加载预训练的分词器和模型
        print("Creat tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(mode_path, add_eos_token=False, add_bos_token=False, eos_token='<eod>')
        self.tokenizer.add_tokens(['<sep>', '<pad>', '<mask>', '<predict>', '<FIM_SUFFIX>', '<FIM_PREFIX>', '<FIM_MIDDLE>','<commit_before>','<commit_msg>','<commit_after>','<jupyter_start>','<jupyter_text>','<jupyter_code>','<jupyter_output>','<empty_output>'], special_tokens=True)

        print("Creat model...")
        self.model = AutoModelForCausalLM.from_pretrained(mode_path, torch_dtype=torch.bfloat16, trust_remote_code=True).cuda()

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        prompt = prompt.strip()
        prompt += "<sep>"
        inputs = self.tokenizer(prompt, return_tensors="pt")["input_ids"].cuda()
        outputs = self.model.generate(inputs,do_sample=False,max_length=4096)
        output = self.tokenizer.decode(outputs[0])
        response = output.split("<sep>")[-1].split("<eod>")[0]

        return response

    @property
    def _llm_type(self) -> str:
        return "Yuan2_LLM"


# 定义一个函数，用于获取llm和embeddings
@st.cache_resource
def get_models():
    llm = Yuan2_LLM(model_path)

    model_kwargs = {'device': 'cuda'}
    encode_kwargs = {'normalize_embeddings': True} # set True to compute cosine similarity
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model_path,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
    )
    return llm, embeddings


summarizer_template = """
假设你是一个AI科研助手，请用一段话概括下面文章的主要内容，200字左右。

{text}
"""


# 定义Summarizer类
class Summarizer:
    """
    class for Summarizer.
    """

    def __init__(self, llm):
        self.llm = llm
        self.prompt = PromptTemplate(
            input_variables=["text"],
            template=summarizer_template
        )
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)

    def summarize(self, docs):
        # 从第一页中获取摘要
        content = docs[0].page_content.split('ABSTRACT')[1].split('KEY WORDS')[0]

        summary = self.chain.run(content)
        return summary


chatbot_template  = '''
假设你是一个AI科研助手，请基于背景，简要回答问题。

背景：
{context}

问题：
{question}
'''.strip()


# 定义ChatBot类
class ChatBot:
    """
    class for ChatBot.
    """

    def __init__(self, llm, embeddings):
        self.prompt = PromptTemplate(
            input_variables=["text"],
            template=chatbot_template
        )
        self.chain = load_qa_chain(llm=llm, chain_type="stuff", prompt=self.prompt)
        self.embeddings = embeddings

        # 加载 text_splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=450,
            chunk_overlap=10,
            length_function=len
        )

    def run(self, docs, query):
        # 读取所有内容
        text = ''.join([doc.page_content for doc in docs])

        # 切分成chunks
        all_chunks = self.text_splitter.split_text(text=text)

        # 转成向量并存储
        VectorStore = FAISS.from_texts(all_chunks, embedding=self.embeddings)

        # 检索相似的chunks
        chunks = VectorStore.similarity_search(query=query, k=1)

        # 生成回复
        response = self.chain.run(input_documents=chunks, question=query)

        return chunks, response


def main():
    # 创建一个标题
    st.title('💬 Yuan2.0 AI科研助手')

    # 获取llm和embeddings
    llm, embeddings = get_models()

    # 初始化summarizer
    summarizer = Summarizer(llm)

    # 初始化ChatBot
    chatbot = ChatBot(llm, embeddings)

    # 上传pdf
    uploaded_file = st.file_uploader("Upload your PDF", type='pdf')

    if uploaded_file:
        # 加载上传PDF的内容
        file_content = uploaded_file.read()

        # 写入临时文件
        temp_file_path = "temp.pdf"
        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(file_content)

        # 加载临时文件中的内容
        loader = PyPDFLoader(temp_file_path)
        docs = loader.load()

        st.chat_message("assistant").write(f"正在生成论文概括，请稍候...")

        # 生成概括
        summary = summarizer.summarize(docs)
        
        # 在聊天界面上显示模型的输出
        st.chat_message("assistant").write(summary)

        # 接收用户问题
        if query := st.text_input("Ask questions about your PDF file"):

            # 检索 + 生成回复
            chunks, response = chatbot.run(docs, query)

            # 在聊天界面上显示模型的输出
            st.chat_message("assistant").write(f"正在检索相关信息，请稍候...")
            st.chat_message("assistant").write(chunks)

            st.chat_message("assistant").write(f"正在生成回复，请稍候...")
            st.chat_message("assistant").write(response)


if __name__ == '__main__':
    main()
