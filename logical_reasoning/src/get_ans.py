import random
import json
import os
import re
from tqdm import tqdm
import uuid
import numpy as np
import requests
from utils.chain import get_chain
import pdb
import errno

MODEL_NAME = 'Qwen2-72B'

chain = get_chain("logical_reasoning")

def extract_from_output(input_text):
    """
    从大模型输出中提取答案。

    参数:
    - input_text: 包含答案的输入文本

    返回:
    - 提取的答案字符
    """

    # 定义正则表达式模式，用于匹配答案
    ans_pattern = re.compile(r"答案是[:：](.)", re.S)

    # 使用正则表达式查找所有匹配的答案
    problems = ans_pattern.findall(input_text)

    # 返回第一个匹配的答案
    return problems[0]

def produce(data, return_list, writer):
    """
    处理任务数据，调用 Qwen API 获取答案，并将结果添加到返回列表中。

    参数:
    - data: 包含任务数据的列表
    - MODEL_NAME: 模型名称
    - return_list: 用于存储结果的列表
    - writer: 文件写入对象

    返回:
    - 无（结果存储在 return_list 中）
    """

    # 使用 tqdm 显示进度条
    tqdm1 = tqdm

    # 遍历任务数据
    for i, task in enumerate(tqdm1(data)): 
        problem = task['problem']  # 获取题目背景信息

        # 遍历每个问题
        for question in task['questions']:
            # pdb.set_trace()
            # 调用大模型获取分析结果
            options = '\n'.join(f"{'ABCDEFG'[i]}. {o}" for i, o in enumerate(question['options']))
            result = chain.invoke({
                "problem": problem,
                "question": question['question'],
                "options": options
            })
            print(f"============>problem: {problem}")
            print(f"============>question: {question['question']}")
            print(f"============>options: \n{options}")
            print(f"============>response: {result}")

            try:
                # 提取响应中的答案
                extract_result = extract_from_output(result)
                question[MODEL_NAME] = extract_result  # 将答案添加到问题中
            except:
                # 如果提取答案失败，继续尝试下一个问题
                pass
        
        # 将处理后的任务添加到返回列表中
        return_list.append(task)

        # 每处理 10 条数据，将结果写入文件
        if (i + 1) % 10 == 0:
            for sample in return_list:
                writer.write(json.dumps(sample, ensure_ascii=False))
                writer.write('\n')
            writer.flush()
            return_list.clear()  # 清空返回列表

def produce_without_eval(data, return_list, writer):
    """
    处理任务数据，调用 Qwen API 获取答案，并将结果添加到返回列表中（用于测试数据）。

    参数:
    - data: 包含任务数据的列表
    - return_list: 用于存储结果的列表
    - writer: 文件写入对象

    返回:
    - 无（结果存储在 return_list 中）
    """

    # 使用 tqdm 显示进度条
    tqdm1 = tqdm

    # 遍历任务数据
    for i, task in enumerate(tqdm1(data)): 
        problem = task['problem']
        task_id = task['id']
        questions = task['questions']
        answers = []

        # 遍历每个问题
        for question in questions:
            # 调用大模型获取分析结果
            options = '\n'.join(f"{'ABCDEFG'[i]}. {o}" for i, o in enumerate(question['options']))
            result = chain.invoke({
                "problem": problem,
                "question": question['question'],
                "options": options
            })
            print(f"============>problem: {problem}")
            print(f"============>question: {question['question']}")
            print(f"============>options: \n{options}")
            print(f"============>response: {result}")

            try:
                extract_result = extract_from_output(result)
                answers.append({'answer': extract_result})
            except Exception as e:
                print(f"Error extracting response: {e}")
                answers.append({'answer': None})
                continue
        
        # 将处理后的任务添加到返回列表中
        return_list.append({'id': task_id, 'questions': answers})

        # 每处理 20 条数据，将结果写入文件
        if (i + 1) % 20 == 0:
            for sample in return_list:
                writer.write(json.dumps(sample, ensure_ascii=False))
                writer.write('\n')
            writer.flush()
            return_list.clear()  # 清空返回列表

    # 处理完所有数据后，将剩余的结果写入文件
    for sample in return_list:
        writer.write(json.dumps(sample, ensure_ascii=False))
        writer.write('\n')
    writer.flush()
    return_list.clear()  # 清空返回列表


def evaluate(ofn):
    """
    评估模型的性能。

    参数:
    - ofn: 包含评估数据的文件名

    返回:
    - 无（结果打印到控制台）
    """
    if not os.path.exists(ofn):
        print(f"文件 {ofn} 不存在")
        return

    data = []

    try:
        with open(ofn) as reader:
            for line in reader:
                sample = json.loads(line)  # 将每行 JSON 数据解析为 Python 字典
                data.append(sample)  # 将解析后的数据添加到列表中
    except json.JSONDecodeError as e:
        print(f"JSON 解析错误: {e}")
        return

    pse = 0  # 记录没有模型预测结果的问题数量
    cnt = 0  # 记录模型预测正确的数量
    tot = 0  # 记录总的问题数量

    for task in data:
        for question in task['questions']:
            if MODEL_NAME in question:
                tot += 1  # 增加总的问题数量
                cnt += question[MODEL_NAME] == question['answer']  # 增加预测正确的数量
            else:
                pse += 1  # 增加没有模型预测结果的问题数量

    print(f"{cnt}, {tot}, {cnt/tot if tot > 0 else 0}, {pse}")  # 打印结果

def process_jsonl_file(ifn):
    data = []
    # 读取 JSONL 文件并将数据转换为列表
    with open(ifn, 'r', encoding='utf-8') as reader:
        for line in reader:
            sample = json.loads(line)
            data.append(sample)
    return data

def get_ans(ifn, ofn, is_train=True):
    """
    主函数，处理 JSONL 文件，提取提示词，并将结果保存到新的文件中。

    参数:
    - ifn: 输入的 JSONL 文件路径
    - ofn: 输出的文件路径

    返回:
    - 无（结果保存到输出文件中）
    """
    # 如果输出文件已经存在，则抛出异常（test.jsonl 为测试专用，允许覆盖）
    if os.path.exists(ofn) and os.path.basename(ofn) != 'test.jsonl':
        raise FileExistsError(f"Output file {ofn} already exists. Aborting processing.")
    
    # 若输出路径文件不存在，则创建文件
    try:
        with open(ofn, 'w', encoding='utf-8') as f:
            pass  # 创建文件
    except OSError as e:
        raise RuntimeError(f"Failed to create output file {ofn}: {e}")
    
    data = process_jsonl_file(ifn)  # 处理 jsonl 文件
    return_list = []

    # 打开输出文件进行写入
    with open(ofn, 'w', encoding='utf-8') as writer:
        # 根据训练数据还是测试数据使用不同的 produce 函数
        if is_train:
            random.shuffle(data)
            produce(data, return_list, writer)
        else:
            produce_without_eval(data, return_list, writer)
        
        # 将剩余的结果写入输出文件
        for sample in return_list:
            writer.write(json.dumps(sample, ensure_ascii=False))
            writer.write('\n')
        
    print("All tasks finished!")
    if is_train:
        evaluate(ofn)
