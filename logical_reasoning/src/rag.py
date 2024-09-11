from typing import List
import numpy as np
import torch
import re
import json
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from collections import Counter
import random
import time
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 定义向量模型类
class EmbeddingModel:
    """
    class for EmbeddingModel
    """
    def __init__(self, path: str) -> None:
        '''
        初始化嵌入模型，加载预训练模型和分词器。
        参数:
            path (str): 预训练模型的路径。'''
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModel.from_pretrained(path).to(device)
        print(f'Loading EmbeddingModel from {path}.')

    def get_embeddings(self, texts: List) -> List[float]:
        """
        计算文本列表的嵌入。
        参数:
            texts (List[str]): 要计算嵌入的文本列表。
        返回:
            List[float]: 输入文本的嵌入列表。
        """
        encoded_input = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt').to(device)  # 对输入文本进行分词，添加填充和截断，并转换为PyTorch张量
        with torch.no_grad():
            model_output = self.model(**encoded_input)  # 将编码后的输入传递给模型，获取模型输出
            sentence_embeddings = model_output[0][:, 0]  # 提取每个输入文本的 [CLS] 标记（第一个标记）的嵌入
        sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)  # 对嵌入进行L2归一化
        return sentence_embeddings.tolist()  # 将嵌入转换为列表并返回
    
# 定义向量库索引类
class VectorStoreIndex:
    """
    向量库索引类，用于存储文档向量并提供查询功能。
    """
    def __init__(self, document_path: str, embed_model: EmbeddingModel) -> None:
        """
        初始化向量库索引类，加载文档并计算嵌入。
        参数:
            document_path (str): 文档文件路径，每行一个文档。
            embed_model (EmbeddingModel): 用于计算嵌入的嵌入模型。
        """
        self.documents = []  # 用于储存知识库文档的内容
        # 读取文档文件中的每一行，并将其添加到文档列表中
        for line in open(document_path, 'r', encoding='utf-8'):
            line = line.strip()
            self.documents.append(line)

        self.embed_model = embed_model
        self.vectors = self.embed_model.get_embeddings(self.documents)  # 计算所有文档的嵌入向量

        print(f'从 {document_path} 加载了 {len(self.documents)} 个文档。')

    def get_similarity(self, vector1: List[float], vector2: List[float]) -> float:
        """
        计算两个向量之间的余弦相似度。
        参数:
            vector1 (List[float]): 第一个向量。
            vector2 (List[float]): 第二个向量。
        返回:
            float: 两个向量之间的余弦相似度。
        """
        dot_product = np.dot(vector1, vector2)  # 计算点积
        magnitude = np.linalg.norm(vector1) * np.linalg.norm(vector2)  # 计算向量的范数（长度）
        if not magnitude:  # 如果范数为零，返回相似度为零
            return 0
        return dot_product / magnitude  # 返回余弦相似度

    def query(self, question: str, k: int = 1) -> List[str]:
        """
        根据查询问题返回最相似的文档。
        参数:
            question (str): 查询问题。
            k (int): 返回最相似的前 k 个文档。
        返回:
            List[str]: 最相似的前 k 个文档。
        """
        question_vector = self.embed_model.get_embeddings([question])[0]  # 计算查询问题的嵌入向量
        result = np.array([self.get_similarity(question_vector, vector) for vector in self.vectors])  # 计算查询向量与所有文档向量之间的相似度
        # print(result)
        return np.array(self.documents)[result.argsort()[-k:][::-1]].tolist()  # 返回相似度最高的前 k 个文档

# 定义向量库索引类
class VectorStoreIndexBatch:
    """
    向量库索引类，用于存储文档向量并提供查询功能。
    """
    def __init__(self, document_path: str, embed_model: EmbeddingModel, batch_size: int = 100) -> None:
        """
        初始化向量库索引类，加载文档并计算嵌入。
        参数:
            document_path (str): 文档文件路径，每行一个文档。
            embed_model (EmbeddingModel): 用于计算嵌入的嵌入模型。
            batch_size (int): 每批次处理的文档数量。
        """
        self.documents = []  # 用于储存知识库文档的内容
        # 读取文档文件中的每一行，并将其添加到文档列表中
        for line in open(document_path, 'r', encoding='utf-8'):
            line = line.strip()
            self.documents.append(line)

        self.embed_model = embed_model
        self.batch_size = batch_size
        self.vectors = self.compute_embeddings_in_batches(self.documents)  # 分批计算所有文档的嵌入向量

        print(f'从 {document_path} 加载了 {len(self.documents)} 个文档。')

    def compute_embeddings_in_batches(self, documents: List[str]) -> List[float]:
        """
        分批计算文档的嵌入向量。
        参数:
            documents (List[str]): 文档列表。
        返回:
            List[float]: 文档的嵌入向量列表。
        """
        vectors = []
        for i in range(0, len(documents), self.batch_size):
            batch_documents = documents[i: i + self.batch_size]
            batch_vectors = self.embed_model.get_embeddings(batch_documents)
            vectors.extend(batch_vectors)
        return vectors

    def get_similarity(self, vector1: List[float], vector2: List[float]) -> float:
        """
        计算两个向量之间的余弦相似度。
        参数:
            vector1 (List[float]): 第一个向量。
            vector2 (List[float]): 第二个向量。
        返回:
            float: 两个向量之间的余弦相似度。
        """
        dot_product = np.dot(vector1, vector2)  # 计算点积
        magnitude = np.linalg.norm(vector1) * np.linalg.norm(vector2)  # 计算向量的范数（长度）
        if not magnitude:  # 如果范数为零，返回相似度为零
            return 0
        return dot_product / magnitude  # 返回余弦相似度

    def query(self, question: str, k: int = 1) -> List[str]:
        """
        根据查询问题返回最相似的文档。
        参数:
            question (str): 查询问题。
            k (int): 返回最相似的前 k 个文档。
        返回:
            List[str]: 最相似的前 k 个文档。
        """
        question_vector = self.embed_model.get_embeddings([question])[0]  # 计算查询问题的嵌入向量
        result = np.array([self.get_similarity(question_vector, vector) for vector in self.vectors])  # 计算查询向量与所有文档向量之间的相似度
        # print(result)
        return np.array(self.documents)[result.argsort()[-k:][::-1]].tolist()  # 返回相似度最高的前 k 个文档

        
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 指定使用的设备

    # 加载 embedding 模型
    embed_model_path = '/data/disk4/home/chenrui/ai-project/logical_reasoning/bge-small-zh-v1___5'  # 还有 large 版本，可以考虑尝试   
    embed_model = EmbeddingModel(embed_model_path)  # 加载 embedding 模型

    # 加载向量知识库
    doecment_path = '/data/disk4/home/chenrui/ai-project/logical_reasoning/data/output/knowledge.txt'
    index = VectorStoreIndex(doecment_path, embed_model) 

    # 加载模型和LoRA权重
    model_path = '/data/disk4/home/chenrui/InternLM2_5-20B-Chat'
    lora_path = '/data/disk4/home/chenrui/ai-project/logical_reasoning/LLaMA-Factory/saves/internlm2.5-chat-20000-mini/lora/sft/checkpoint-500'  # 这里改称你的 lora 输出对应 checkpoint 地址
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True).eval()
    model = PeftModel.from_pretrained(model, model_id=lora_path) 

    question = '\n你是一个逻辑推理专家，擅长解决逻辑推理问题。以下是一个逻辑推理的题目，形式为单项选择题。所有的问题都是（close-world assumption）闭世界假设，即未观测事实都为假。请逐步分析问题，最终只输出答案对应的选项字母，如\"A\"。题目如下：\n\n### 题目:\n在一个办公室里，有四位员工：Alice, Bob, Charlie和Diana，他们每个人都有着不同的工作习惯和表现。以下是每个员工的具体属性：\n\n- Alice, Diana是女性；Bob, Charlie是男性。\n- 关于工作效率，Diana和Charlie的工作效率是高，Alice的工作效率是中等，Bob的工作效率是低。\n- 关于团队合作，Diana和Alice的团队合作是好，Charlie的团队合作是一般，Bob的团队合作是差。\n- 关于创新能力，Diana和Alice的创新能力是强，Charlie的创新能力是一般，Bob的创新能力是弱。\n- 关于客户满意度，Diana和Alice的客户满意度是高，Bob的客户满意度是中等，Charlie的客户满意度是低。\n\n根据以上信息，请回答以下单项选择题：\n\n### 问题:\n选择题 3：\n哪些员工被认为是创新型员工（创新能力和工作效率均为中等或高）？\nA. Alice和Diana\nB. Bob和Charlie\nC. Charlie和Diana\nD. Alice和Bob\n'
    print('> Question:', question)

    context = index.query(question)
    print('> Context:', context)  

    prompt_rag = f'背景：{context}\n问题：{question}\n请基于背景，回答问题。'
    prompt = question

    def llm_generate(prompt):
        messages = [
            {"role": "user", "content": prompt}
        ]
        inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_tensors="pt", return_dict=True).to(device)

        gen_kwargs = {"max_length": 2500, "do_sample": True, "top_k": 1}

        outputs = model.generate(**inputs, **gen_kwargs)
        outputs = outputs[:, inputs['input_ids'].shape[1]:]
        output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return output
    print(f"With rag: {llm_generate(prompt_rag)}")
    print(f"Without rag: {llm_generate(prompt)}")


