import random
import json
import os
import re
from tqdm import tqdm
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils.chain import get_chain

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

def produce(data, return_list, lock, pbar, ofn):
    """
    处理任务数据，调用 Qwen API 获取答案，并将结果添加到返回列表中。

    参数:
    - data: 包含任务数据的列表
    - MODEL_NAME: 模型名称
    - return_list: 用于存储结果的列表
    - lock: 用于文件写入操作的线程锁
    - pbar: 总进度条对象

    返回:
    - 无（结果存储在 return_list 中）
    """

    # 使用 tqdm 显示进度条
    tqdm1 = tqdm

    batch_size = 50
    batch = []

    # 遍历任务数据
    for i, task in enumerate(data): 
        problem = task['problem']  # 获取题目背景信息

        # 遍历每个问题
        for question in task['questions']:
            # 调用大模型获取分析结果
            try:
                options = '\n'.join(f"{'ABCDEFG'[i]}. {o}" for i, o in enumerate(question['options']))
                result = chain.invoke({
                    "problem": problem,
                    "question": question['question'],
                    "options": options
                })
                # print(f"============>problem: {problem}")
                # print(f"============>question: {question['question']}")
                # print(f"============>options: \n{options}")
                # print(f"============>response: {result}")
            except:
                pass

            try:
                # 提取响应中的答案
                extract_result = extract_from_output(result)
                question[MODEL_NAME] = extract_result  # 将答案添加到问题中
            except:
                # 如果提取答案失败，继续尝试下一个问题
                pass
        
        # 将处理后的任务添加到返回列表中
        return_list.append(task)
        batch.append(task)

        # 如果达到了批次大小或者已经是最后一个元素，则进行写入
        if len(batch) >= batch_size or i == len(data) - 1:
            # 写入数据
            with lock:
                # 写入文件
                for sample in batch:
                    with open(ofn, 'a', encoding='utf-8') as writer:
                        writer.write(json.dumps(sample, ensure_ascii=False) + '\n')
            
            # 清空临时列表
            batch = []

        # 更新进度条
        pbar.update(1)

    # 输出任务完成的信息
    print(f"Task {threading.current_thread().name} completed.")


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
    - is_train: 是否为训练数据

    返回:
    - 无（结果保存到输出文件中）
    """
    # 如果输出文件已经存在，则创建一个新的文件名
    if os.path.exists(ofn):
        base_name, ext = os.path.splitext(ofn)
        counter = 1
        while True:
            new_ofn = f"{base_name}_{counter}{ext}"
            if not os.path.exists(new_ofn):
                ofn = new_ofn
                break
            counter += 1
    
    # 若输出路径文件不存在，则创建文件
    try:
        with open(ofn, 'w', encoding='utf-8') as f:
            pass  # 创建文件
    except OSError as e:
        raise RuntimeError(f"Failed to create output file {ofn}: {e}")
    
    data = process_jsonl_file(ifn)  # 处理 jsonl 文件
    return_list = []

    # 创建一个锁来确保线程安全地写入文件
    lock = threading.Lock()

    # 创建一个总体进度条
    total_tasks = len(data)
    pbar = tqdm(total=total_tasks, desc="Overall Progress", leave=True)

    # 打开输出文件进行写入
    with open(ofn, 'w', encoding='utf-8') as writer:
        random.shuffle(data)
        # 使用 ThreadPoolExecutor 进行并行处理
        with ThreadPoolExecutor(max_workers=100) as executor:
            # 提交任务
            num_tasks = 260
            futures = [executor.submit(produce, data[i:i+num_tasks], return_list, lock, pbar, ofn) for i in range(0, len(data), num_tasks)]  # 设置线程数和每个线程处理数量
            # 等待所有任务完成
            for future in as_completed(futures):
                future.result()

        # 在所有线程完成后写入文件
        # with lock:
        #     for sample in return_list:
        #         writer.write(json.dumps(sample, ensure_ascii=False))
        #         writer.write('\n')
        #     writer.flush()

    print("All tasks finished!")
    if is_train:
        evaluate(ofn)

# 示例调用
if __name__ == "__main__":
    input_file = "/path/to/input.jsonl"
    output_file = "/path/to/output.jsonl"
    get_ans(input_file, output_file, is_train=True)