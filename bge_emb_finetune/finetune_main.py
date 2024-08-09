from embedding_finetune.data_process.get_train_test_jsonl import *
from embedding_finetune.evaluate.create_index import get_corpus_list,store_to_db
from embedding_finetune.evaluate.evaluate_precise import get_es_db,search,evaluate
from embedding_finetune.utils import (
    ACGE,
    ACGE_finetune,
    BGE_EMB,
    BGE_EMB_finetune
)
import subprocess
from tqdm import tqdm
from typing import List

def get_all_label(label_path:str):
    '''
    获得所有的疾病标签
    input:
        label_path:疾病txt文件路径
    '''
    with open(label_path,"r") as file:
        all_label = file.read()
    all_label_list = all_label.split('\n')
    #去重
    all_label_list = list(set(all_label_list))
    print("===总标签个数为===",len(all_label_list))
    return all_label_list

def save_train_test_data(file_path:str,all_label_list:List[str],finetune_json_path:str,eval_json_path:str):
    '''
    获得训练数据和测试数据
    训练数据集有pos和neg
    测试数据集只有pos即可
    input:
        file_path:原始文件路径
        all_label_list:所有疾病标签的列表
        finetune_json_path:需要保存的微调数据集路径
        eval_json_path:需要保存的测试集路径
    '''
    train_df, test_df = process_csv(file_path,all_label_list)
    #获得数据jsonl
    get_pos_neg_json(finetune_json_path,train_df,all_label_list)
    get_pos_json(eval_json_path,test_df,all_label_list)
    print("===获得微调数据和测试数据===")

def run_mine_HD_and_finetune_sh(model_type:str):
    '''
    执行挖掘硬负样本和微调,需要修改一下sh对应的参数!!!
    input:
        model_type:模型类型,主要为了标识bge和acge,方便可以定位到特定的sh文件而已
    '''
    #运行挖掘hard negatives sh
    mine_HD_sh_path = 'hard_negatives_{}.sh'.format(model_type)
    subprocess.run(['bash', mine_HD_sh_path])
    print("===挖掘硬负样本成功===")

    #运行微调sh
    finetuen_path_sh_path = 'finetune_{}.sh'.format(model_type)
    subprocess.run(['bash', finetuen_path_sh_path])
    print("===微调成功===")

def create_corpus_index(index_name:str,corpus_file:str,embedding_model):
    '''
    将所有疾病label创建一个ElasticsearchStore数据库
    input:
        index_name:建立数据库的名称
        corpus_file:疾病label的json路径
        embedding:就是要实例化的要测试的模型
    '''
    corpus = get_corpus_list(corpus_file)
    store_to_db(corpus,index_name,embedding_model)
    print('===创建index成功===')

def cal_precise(index_name:str,eval_json_path:str,embedding_model):
    '''
    计算准确率
    input:
        index_name:建立数据库的名称
        eval_json_path:测试数据集的路径
        embedding:就是要实例化的要测试的模型
    '''
    #计算准确率
    retrieval_result_list = []
    ground_truth_list = []
    db = get_es_db(index_name,embedding_model)
    for line in tqdm(open(eval_json_path)):
        query_dic = json.loads(line.strip())
        query = query_dic['query']
        retrieval_result = search(query,db,k=5)
        ground_truth = query_dic['pos']

        retrieval_result_list.append(retrieval_result)
        ground_truth_list.append(ground_truth)

    metrics = evaluate(preds=retrieval_result_list, labels=ground_truth_list, cutoffs=[1,2,5])
    return metrics



if __name__ == '__main__':
    # model_type = 'bge'
    model_type = 'acge'

    #得到所有的疾病类型
    label_path = './embedding_finetune/data/label.txt'
    all_label_list = get_all_label(label_path)

    #获得数据
    file_path = './embedding_finetune/data/disease.xlsx'
    finetune_json_path = "./embedding_finetune/data_process/finetune_data.jsonl"
    eval_json_path = "./embedding_finetune/data_process/evaluate.jsonl"
    save_train_test_data(file_path,all_label_list,finetune_json_path,eval_json_path)

    #挖掘Hard negative 和微调
    run_mine_HD_and_finetune_sh(model_type)

    ###############################
    index_name = 'test_pipeline6'
    corpus_file = './embedding_finetune/data/label.json'
    model_path = './output/{}_finetune'.format(model_type)
    query_instruction_for_retrieval = ''
    
    if model_type == 'acge':
        # embedding_model = ACGE.ACGEEmbedding(model_path)
        embedding_model = ACGE_finetune.ACGEEmbedding_finetune(model_path)
    if model_type == 'bge':
        # embedding_model = BGE_EMB.BGEEmbedding(model_path,query_instruction_for_retrieval)
        embedding_model = BGE_EMB_finetune.BGEEmbedding_finetune(model_path,query_instruction_for_retrieval)

    #创建index
    create_corpus_index(index_name,corpus_file,embedding_model)

    #计算准确率
    metrics = cal_precise(index_name,eval_json_path,embedding_model)
    print('准确率：',metrics)




