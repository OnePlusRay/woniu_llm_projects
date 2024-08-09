#对得到的excel文件做特殊处理
import pandas as pd
import numpy as np
import re

# 定义一个函数来筛选掉每行第一个汉字前的所有内容
def remove_before_first_chinese(text):
    # 使用正则表达式匹配第一个汉字及其之前的所有内容
    match = re.search(r'[\u4e00-\u9fff]', text)
    if match:
        return text[match.start():]
    return text

def get_special_process_excel(input_file):
    df = pd.read_excel(input_file)
    df = df[~pd.isna(df['Question'])]
    # 筛选掉每行第一个汉字前的所有内容
    df['Question'] = df['Question'].apply(remove_before_first_chinese)
    # 删除指定列中的所有'/n/n'
    df['Question'] = df['Question'].str.replace('/n/n', '')
    df.to_excel(input_file,index=False)
    return 

if __name__ == '__main__':
    input_file = '../data/new_disease.xlsx'
    get_special_process_excel(input_file)