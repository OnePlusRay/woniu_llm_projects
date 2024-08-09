import os
import json
import sys
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import ElasticsearchStore
from langchain_openai.embeddings.base import OpenAIEmbeddings
from langchain_openai import AzureOpenAIEmbeddings
from elasticsearch import Elasticsearch


current_dir = os.path.dirname(os.path.abspath(__file__))
utils_dir = os.path.join(current_dir, '..', 'utils')
sys.path.append(utils_dir)
from BGE_EMB import BGEEmbedding
from BGE_EMB_finetune import BGEEmbedding_finetune
from ACGE import ACGEEmbedding
from ACGE_finetune import ACGEEmbedding_finetune

from dotenv import load_dotenv
load_dotenv()

def get_corpus_list(input_file):
    corpus = []
    for line in open(input_file):
        line = json.loads(line.strip())
        corpus.append(line['content'])
    return corpus

def store_to_db(corpus,index_name,embeddings):
    text_splitter = CharacterTextSplitter(chunk_size=50, chunk_overlap=0) #chunk_size表示分割块的大小，chunk_overlap表示分割块之间重叠部分的大小
    docs = text_splitter.create_documents(corpus) #返回一个包含分割后文本块的列表docs

    es_url = list(os.getenv("ES_URL").split(","))
    connection = Elasticsearch(
        es_url, 
        ca_certs = "./http_ca.crt", 
        verify_certs = True, 
        http_auth=(os.getenv("ES_USER"), os.getenv("ES_PASSWORD"))
        )

    print('正在存入向量数据库，时间较久请耐心等待')
    db = ElasticsearchStore.from_documents(
        docs,
        embedding = embeddings,
        es_url=es_url,
        index_name=index_name,
        strategy=ElasticsearchStore.ApproxRetrievalStrategy(hybrid=True),
        es_connection = connection,
        es_user=os.getenv("ES_USER"),
        es_password=os.getenv("ES_PASSWORD"),
    )
    db.client.indices.refresh(index=index_name)
    print("更新知识库向量数据库成功")

if __name__ == '__main__':

    # embeddings = AzureOpenAIEmbeddings(
    #     deployment="text-embedding-ada-002",
    #     openai_api_key='b2753222b7ae4110bb3ef10835085c3e',
    #     azure_endpoint='https://insnailgpt4us.openai.azure.com/',
    #     openai_api_version='2024-02-01'
    # )
    # embeddings = AzureOpenAIEmbeddings(
    #     deployment="text-embedding-3-small",
    #     openai_api_key='b2753222b7ae4110bb3ef10835085c3e',
    #     azure_endpoint='https://insnailgpt4us.openai.azure.com/',
    #     openai_api_version='2024-02-01'
    # )
    # embeddings = AzureOpenAIEmbeddings(
    #     deployment="text-embedding-3-large",
    #     openai_api_key='b2753222b7ae4110bb3ef10835085c3e',
    #     azure_endpoint='https://insnailgpt4us.openai.azure.com/',
    #     openai_api_version='2024-02-01'
    # )
    model_path = '../../bge-large-zh-v1.5'
    query_instruction_for_retrieval = ''
    embeddings = BGEEmbedding(model_path,query_instruction_for_retrieval)

    input_file = '../data/label.json'
    index_name = 'acge_test_fine1'
    corpus = get_corpus_list(input_file)
    store_to_db(corpus,index_name,embeddings)
    
