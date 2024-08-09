import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split


def process_csv(file_path,all_label_list):
    #读取原始数据df
    df = pd.read_excel(file_path)
    df = df[~pd.isna(df['Question'])]
    df.dropna(subset=['Label1','Label2','Label3','Label4','Label5'],how='all',inplace=True)
 
    to_check_col = ['Label1', 'Label2', 'Label3','Label4','Label5']

    def check_values(row):
        for each_col in to_check_col:
            if row[each_col] not in all_label_list:
                row[each_col] = np.nan
        return row
    
    df = df.apply(check_values, axis=1)
    df.dropna(subset=['Label1','Label2','Label3','Label4','Label5'],how='all',inplace=True)

    #划分训练集和测试集
    train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)
    print('train_df.shape:',train_df.shape)
    print('test_df.shape',test_df.shape)

    return train_df, test_df

def get_pos_neg_json(json_path,df,all_label_list):
    # 微调数据集{'query':,'pos':,'neg':}
    with open(json_path, "w",encoding='utf-8') as file:
        # 遍历DataFrame的每一行
        for index, row in df.iterrows():
            # 创建一个新的字典
            data_dict = {}
            # 将问题列的值作为键，问题列对应的字符串作为值
            data_dict["query"] = row[df.columns.to_list()[0]]
            # 将label列中不为None的值都放到字典键为"pos"的值中
            pos_values = [row[col] for col in df.columns.to_list()[2:] if not pd.isna(row[col])]
            data_dict["pos"] = pos_values
            data_dict['neg'] = [item for item in all_label_list if item not in pos_values]
            # 将字典写入JSON文件
            json.dump(data_dict, file,ensure_ascii=False)
            # 写入换行符，以便每个字典占据一行
            file.write('\n')

def get_pos_json(json_path,df,all_label_list):
    #测试数据集{'query':,'pos':}
    with open(json_path, "w",encoding='utf-8') as file:
        # 遍历DataFrame的每一行
        for index, row in df.iterrows():
            # 创建一个新的字典
            data_dict = {}
            # 将问题列的值作为键，问题列对应的字符串作为值
            data_dict["query"] = row[df.columns.to_list()[0]]
            # 将label列中不为None的值都放到字典键为"pos"的值中
            pos_values = [row[col] for col in df.columns.to_list()[2:] if not pd.isna(row[col])]
            data_dict["pos"] = pos_values
            # data_dict['neg'] = [item for item in all_label_list if item not in pos_values]
            # 将字典写入JSON文件
            json.dump(data_dict, file,ensure_ascii=False)
            # 写入换行符，以便每个字典占据一行
            file.write('\n')

if __name__ == "__main__":
    #得到所有的疾病类型
    with open("../data/label.txt","r") as file:
        all_label = file.read()
    all_label_list = all_label.split('\n')
    #去重
    all_label_list = list(set(all_label_list))
    print("总标签个数为:",len(all_label_list))

    file_path = '../data/disease.xlsx'
    train_df, test_df = process_csv(file_path,all_label_list)

    #获得数据jsonl
    finetune_json_path = "finetune_data.jsonl"
    eval_json_path = "evaluate.jsonl"
    get_pos_neg_json(finetune_json_path,train_df,all_label_list)
    # get_pos_json('train.jsonl',train_df,all_label_list)
    get_pos_json(eval_json_path,test_df,all_label_list)





