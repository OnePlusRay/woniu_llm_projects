import json
import os
import pdb
import re
import sys
from tqdm import tqdm
from utils.chain import get_chain

chain = get_chain("generate_questions")

def process_jsonl_file(ifn):
    data = []
    # 读取 JSONL 文件并将数据转换为列表
    with open(ifn, 'r', encoding='utf-8') as reader:
        for line in reader:
            sample = json.loads(line)
            data.append(sample)
    return data

def format_generated_string(generated_str):
    """格式化生成的字符串为标准 JSON 格式"""

    # 去除多余的空格和换行符
    formatted_str = re.sub(r'\s+', ' ', generated_str)  # 去除多余空格
    formatted_str = formatted_str.strip()  # 去除首尾空格

    return formatted_str

def generate_questions(ifn, ofn):
    """
    生成问题并将结果写入 JSONL 文件
    :param ifn: 输入文件路径
    :param ofn: 输出文件路径
    """

    # 清空输出文件内容
    if os.path.exists(ofn):
        with open(ofn, 'w', encoding='utf-8') as f:
            f.write('')

    # 使用 tqdm 显示进度条
    tqdm1 = tqdm

    # 加载 jsonl 数据
    data = process_jsonl_file(ifn)

    new_data_list = []
    batch_size = 10

    # 遍历任务数据
    for i, problem in enumerate(tqdm1(data)): 
        result = chain.invoke({
                "problem": problem 
            })
        
        # 格式化生成的内容
        formatted_result = format_generated_string(result)
        # pdb.set_trace()
        
        # 将格式化后的字符串解析为 JSON 对象
        try:
            json_object = json.loads(formatted_result)
            new_data_list.append(json_object)
            print(json_object)
        
            # 每生成 batch_size 条数据写入一次
            if (i + 1) % batch_size == 0:
                with open(ofn, 'a', encoding='utf-8') as writer:
                    for item in new_data_list:
                        writer.write(json.dumps(item, ensure_ascii=False) + '\n')
                    new_data_list.clear()  # 清空缓存区
                print(f"Generated {i + 1} problems, results written to {ofn}")

        except json.JSONDecodeError as e:
            print("JSON 解析错误:", e)
        
    # 将剩余的内容写入 JSONL 文件
    with open(ofn, 'a', encoding='utf-8') as writer:
        for item in new_data_list:
            writer.write(json.dumps(item, ensure_ascii=False) + '\n')  # 每个 JSON 对象占一行

    return new_data_list

if __name__ == "__main__":
    ifn = '/data/disk4/home/chenrui/ai-project/logical_reasoning/data/input/test.jsonl'
    ofn = '/data/disk4/home/chenrui/ai-project/logical_reasoning/data/input/generated_questions.jsonl'  # 输出文件路径
    new_data_list = generate_questions(ifn, ofn)



        
