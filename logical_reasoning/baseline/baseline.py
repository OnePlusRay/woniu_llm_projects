from multiprocessing import Process, Manager
import json
import os
from pprint import pprint
import re
from tqdm import tqdm
import random
import uuid
import openai
import tiktoken
import json
import numpy as np
import requests
from retry import retry
from scipy import sparse
#from rank_bm25 import BM25Okapi
#import jieba
from http import HTTPStatus
import dashscope
import pdb

MODEL_NAME = 'qwen1.5-32b-chat'  # 模型名称（全局变量）

# 使用 retry 装饰器，设置重试次数为 3 次，每次重试之间的延迟为 3 秒。
# 如果函数在调用过程中抛出异常，装饰器会自动重试指定的次数。
@retry(delay=3, tries=3)

def call_qwen_api(MODEL_NAME, query):
    """
    调用 Qwen API 并处理响应。

    参数:
    - MODEL_NAME: 模型名称
    - query: 用户查询内容

    返回:
    - API 响应的内容，如果请求成功
    - 如果请求失败，抛出异常
    """
    # 构建消息列表
    messages = [
        {'role': 'user', 'content': query}]
    
    # 调用 Qwen API
    response = dashscope.Generation.call(
        MODEL_NAME,
        messages=messages,
        result_format='message',  # set the result is message format.
    )

    # 检查响应状态码
    if response.status_code == HTTPStatus.OK:
        print(response)
        return response['output']['choices'][0]['message']['content']
    else:
        # 打印错误信息
        print('Request id: %s, Status code: %s, error code: %s, error message: %s' % (
            response.request_id, response.status_code,
            response.code, response.message
        ))
        # 抛出异常以触发重试
        raise Exception()


def get_prompt(problem, question, options):
    '''构建一个逻辑推理问题的提示文本'''
    # 将选项列表转换为带有字母标签的字符串，每个选项一行，字母标签从 'A' 到 'G'，取决于有多少个选项。
    # 使用列表推导式和str.join()将转换后的选项合并成一个字符串。
    options = '\n'.join(f"{'ABCDEFG'[i]}. {o}" for i, o in enumerate(options))

    prompt = f"""
    你是一个逻辑推理专家，擅长解决逻辑推理问题。以下是一个逻辑推理的题目，形式为单项选择题。
    所有的问题都是（close-world assumption）闭世界假设，即未观测事实都为假。
    请逐步分析问题并在最后一行输出答案，最后一行的格式为"答案是：A"。题目如下：

    ### 题目:
    {problem}

    ### 问题:
    {question}
    {options}
    """

    return prompt


def extract(input_text):
    """
    从输入文本中提取答案。

    参数:
    - input_text: 包含答案的输入文本

    返回:
    - 提取的答案字符
    """

    # 定义正则表达式模式，用于匹配答案
    ans_pattern = re.compile(r"答案是：(.)", re.S)

    # 使用正则表达式查找所有匹配的答案
    problems = ans_pattern.findall(input_text)

    # 返回第一个匹配的答案
    return problems[0]


def produce(data, MODEL_NAME, return_list, pid):
    """
    处理任务数据，调用 Qwen API 获取答案，并将结果添加到返回列表中。

    参数:
    - data: 包含任务数据的列表
    - MODEL_NAME: 模型名称
    - return_list: 用于存储结果的列表
    - pid: 进程 ID，用于控制输出

    返回:
    - 无（结果存储在 return_list 中）
    """

    # 使用 tqdm 显示进度条
    tqdm1 = tqdm

    # 遍历任务数据
    for task in tqdm1(data):
        problem = task['problem']  # 获取题目背景信息

        # 遍历每个问题
        for question in task['questions']:
            # 生成提示文本
            prompt = get_prompt(problem, 
                                question['question'], 
                                question['options'])

            # 调用 Qwen API 获取响应
            response = call_qwen_api(MODEL_NAME, prompt)

            try:
                # 提取响应中的答案
                extract_response = extract(response)
                question[MODEL_NAME] = extract_response  # 将答案添加到问题中

                # 如果进程 ID 为 0，打印答案
                if pid == 0:
                    pprint(extract_response)
                
                # 成功提取答案后，跳出循环
                break
            except:
                # 如果提取答案失败，继续尝试下一个问题
                pass
        
        # 将处理后的任务添加到返回列表中
        return_list.append(task)

def main(ifn, ofn):
    if os.path.exists(ofn):
        pass

    POOL_SIZE = 5
    data = []
    with open(ifn) as reader:
        for line in reader:
            sample = json.loads(line)
            data.append(sample)

    random.shuffle(data)

    datas = [data[i::POOL_SIZE] for i in range(POOL_SIZE)]
    
    with Manager() as manager:
        producers = []
        return_list = manager.list()
        for i in range(POOL_SIZE):
            p = Process(target=produce,
                    args=(datas[i],
                        MODEL_NAME,
                        return_list,
                        i,
                        )
                    )
            producers.append(p)

        for p in producers:
            p.start()

        for p in producers:
            p.join()

        print(len(return_list))

        with open(ofn, 'w') as writer:
            for sample in return_list:
                writer.write(json.dumps(sample, ensure_ascii=False))
                writer.write('\n')
        
    print("All tasks finished!")
    evaluate(ofn)


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

    print(cnt, tot, cnt/tot if tot > 0 else 0, pse)

    

if __name__ == '__main__':

    a = extract("""根据欧几里得算法，逐步解析计算两个数6和7的最大公约数（gcd）的步骤如下：

1. 判断6和7是否相等：不相等。
2. 判断6和7大小关系，7 > 6，所以用更大的数7减去较小的数6得到结果1。
3. 现在计算6和1的最大公约数。
4. 6 > 1，根据算法用更大的数6减去较小的数1得到结果5。
5. 再计算5和1的最大公约数。
6. 5 > 1，用5减去1得到结果4。
7. 再计算4和1的最大公约数。
8. 4 > 1，用4减去1得到结果3。
9. 再计算3和1的最大公约数。
10. 3 > 1，用3减去1得到结果2。
11. 再计算2和1的最大公约数。
12. 2 > 1，用2减去1得到结果1。
13. 最后计算1和1的最大公约数，两数相等，gcd即为这两个数，也就是1。

因此，6和7的最大公约数是1。

答案是：C.""")

    print(a)
    main('/sdata/chenrui/ai-project/Datawhale-ai/data/round1_train_data.jsonl', '/sdata/chenrui/ai-project/Datawhale-ai/data/qwen.jsonl')
