import os
import numpy as np
import json
import sys
from tqdm import tqdm
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import ElasticsearchStore
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


def get_es_db(index_name,embeddings):
    es_url = list(os.getenv("ES_URL").split(","))
    connection = Elasticsearch(es_url, 
                               ca_certs = "./http_ca.crt", 
                               verify_certs = True, 
                               http_auth=(os.getenv("ES_USER"), os.getenv("ES_PASSWORD"))
                               )
    return ElasticsearchStore(
        embedding=embeddings,
        es_user=os.getenv("ES_USER"),
        es_password=os.getenv("ES_PASSWORD"),
        es_url=es_url,
        index_name=index_name,
        es_connection = connection
        )

def evaluate(preds, labels, cutoffs=[1,2,5]):
    """
    Evaluate precise at cutoffs.
    """
    metrics = {}
    precises = np.zeros(len(cutoffs))
    for pred, label in zip(preds, labels):
        label_set = set(label)
        for k,cutoff in enumerate(cutoffs):

            pred_set = set(pred[:cutoff])
            if label_set.issubset(pred_set):
                precises[k] += 1

    precises /= len(preds)
    for i, cutoff in enumerate(cutoffs):
        precise = precises[i]
        metrics[f"precise@{cutoff}"] = precise

    return metrics

def search(query,db,k):
    sim_response = db.similarity_search_with_score(query,k)
    res = []
    for each in sim_response:
        res.append(each[0].page_content)
    return res
    

if __name__ == '__main__':
    retrieval_result_list = []
    ground_truth_list = []

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

    index_name = 'acge_test_fine1'
    model_path = '../../bge-large-zh-v1.5'
    query_instruction_for_retrieval = ''
    embeddings = BGEEmbedding(model_path,query_instruction_for_retrieval)
    db = get_es_db(index_name,embeddings)


    query_file = '../data/query.jsonl'
    for line in tqdm(open(query_file)):
        query_dic = json.loads(line.strip())
        query = query_dic['query']
        retrieval_result = search(query,db,k=5)
        ground_truth = query_dic['pos']

        retrieval_result_list.append(retrieval_result)
        ground_truth_list.append(ground_truth)

    # print("retrieval_result_list:",retrieval_result_list)
    # print("ground_truth_list:",ground_truth_list)
    metrics = evaluate(preds=retrieval_result_list, labels=ground_truth_list, cutoffs=[1,2,5])
    print('准确率：',metrics)

