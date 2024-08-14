import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import re
import pdb
import re
import time
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from utils.chain import get_chain
import random
import concurrent.futures

chain = get_chain('generate_problems')  # 需要选择 gpt4

   
def write_to_output_file(item, ofn):
    '''写入数据到输出文件'''
    try:
        with open(ofn, 'a', encoding='utf-8') as f:  # 以追加模式打开文件
            f.write(f"{json.dumps(item, ensure_ascii=False)}\n")
    except Exception as e:
        print(f"写入输出文件时发生错误: {e}")

def process_batch(i, ifn):
    data = []
    
    try:
        with open(ifn, 'r', encoding='utf-8') as f:
            lines = f.readlines()  # 读取所有行

        # 随机抽取五条非空行
        non_empty_lines = [line.strip() for line in lines if line.strip()]  # 过滤非空行
        random_lines = random.sample(non_empty_lines, min(5, len(non_empty_lines)))  # 随机抽取五条
        example = [{"problem": "考虑两个正整数 X 和 Y，我们想找到其最大公约数（GCD）。以下是一些特定情况的最大公约数查询：", "questions": [{"question": "选择题 1：\n当两个数分别是 7 和 4，它们的最大公约数是多少？", "options": ["1", "2", "3", "4"], "answer": "A"}, {"question": "选择题 2：\n当两个数分别是 8 和 2，它们的最大公约数是多少？", "options": ["2", "4", "6", "8"], "answer": "A"}, {"question": "选择题 3：\n当两个数分别是 7 和 4，是否可能它们的最大公约数是 4？", "options": ["是", "否"], "answer": "B"}, {"question": "选择题 4：\n当两个数分别是 8 和 4，是否可能它们的最大公约数是 6？", "options": ["是", "否"], "answer": "B"}], "id": "round1_train_data_031"}]

        for index, line in enumerate(random_lines):
            try:
                data.append(json.loads(line))  # 解析JSON并添加到data中
            except json.JSONDecodeError as e:
                print(f"行 {index} JSON解析错误: {e}，跳过该行。")
        res = chain.invoke({
            "example": example,
            "content": data
        })
        
        start = res.find('{')
        end = res.rfind('}')
        
        if start != -1 and end != -1 and start < end:
            new_string = res[start:end + 1]  # 提取从第一个左中括号到最后一个右中括号的内容
            print(f"=============>得到结果：{new_string}")
            new_string = new_string.replace("'", '"')

            return new_string  # 返回结果以便后续处理
        else:
            print("未找到有效的JSON字符串，跳过该结果。")
            return None
    
    except Exception as e:
        print(f"处理第 {i} 批次时发生错误: {e}")
        return None
    
def generate_problems_con(ifn, ofn):
    '''多线程生成数据'''
    output_data = []
    num_problems = 2000  # 生成数据的条数
    num_threads = 10

    # 若ofn存在，则清空输出文件内容
    if os.path.exists(ofn):
        with open(ofn, 'w', encoding='utf-8') as f: 
            f.write('')

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:  # 这里每个线程的任务数量分配是自动进行的
        futures = [executor.submit(process_batch, i, ifn) for i in range(num_problems)]  
        
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            result = future.result()
            if result:
                output_data.append(result)

            # 每 20 次输出后，将列表中的内容写入文件
            if (i + 1) % 20 == 0:
                for item in output_data:
                    try:
                        parsed_data = json.loads(item)
                        write_to_output_file(parsed_data, ofn)  # 直接写入输出文件
                    except json.JSONDecodeError as e:
                        print(f"解析错误: {e}，跳过该字符串。")

                output_data.clear()  # 清空列表以便下次使用

                time.sleep(3)  # 等待3秒

    # 处理完所有批次后，确保写入剩余的数据
    for item in output_data:
        try:
            parsed_data = json.loads(item)
            write_to_output_file(parsed_data, ofn)  # 直接写入输出文件
        except json.JSONDecodeError as e:
            print(f"解析错误: {e}，跳过该字符串。")

    print(f"Data has been written to {ofn}")

if __name__ == "__main__":
    # 输入和输出路径
    ifn = '/data/disk4/home/chenrui/ai-project/logical_reasoning/data/input/new/train_data_new_2.jsonl'
    ofn = './data/output/new_data_gpt4.jsonl'
