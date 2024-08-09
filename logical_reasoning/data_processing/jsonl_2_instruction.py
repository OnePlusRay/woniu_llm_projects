import os
import json
import shutil
from pprint import pprint
from utils import get_prompt, get_prompt_analysis, get_prompt_score, split_dataset, get_prompt_complex


# def get_prompt(problem, question, options):
#     '''构建提示词模板'''
#     options = '\n'.join(f"{'ABCDEFG'[i]}. {o}" for i, o in enumerate(options))
#     prompt = f"""你是一个逻辑推理专家，擅长解决逻辑推理问题。以下是一个逻辑推理的题目，形式为单项选择题。所有的问题都是（close-world assumption）闭世界假设，即未观测事实都为假。请逐步分析问题，最终只输出答案对应的选项字母，如"A"。题目如下：

# 你需要按照下面的步骤来完成任务：
# 1. 分析并理解题目内容
# 2. 逐步对问题的各个选项进行分析，选择出最可能的答案
# 3. 输出上一步中答案对应的选项字母

# 题目:
# {problem}

# 问题:
# {question}
# {options}
# """
#     return prompt

def construct_prompt(problem, question, options, answer, task_type):
    if task_type == 'analysis':
        instruction = get_prompt_analysis(problem, question, options)
    elif task_type == 'score':
        instruction = get_prompt_score(problem, question, options)
    elif task_type == 'raw':
        instruction = get_prompt(problem, question, options)
    elif task_type == 'complex':
        instruction = get_prompt_complex(problem, question, options)
    output = answer
    return {
        "instruction": instruction,
        "input": "",
        "output": output
    }

def generate_instruction_dataset(input_file, output_file, tmp_file, task_type, need_split=True):
    '''原 JSONL 数据 -> 指令集数据'''
    if need_split:
        input_file = split_dataset(input_file, tmp_file)  # 调用切分原 JSONL 的子问题的函数（增加重写 id 操作，可以调整新 id 从哪开始生成）
    converted_data = []

    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            problem = item["problem"]
            question = item["question"]
            options = item["options"]
            answer = item["answer"]
            converted_item = construct_prompt(problem, question, options, answer, task_type)
            converted_data.append(converted_item)

    # 将转换后的数据集输出为JSON格式
    json_output = json.dumps(converted_data, ensure_ascii=False, indent=2)

    # 打印JSON输出
    print(json_output)

    # 保存JSON到文件
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(json_output)



if __name__ == '__main__':

    # 读取JSONL文件
    input_file = '/data/disk4/home/chenrui/ai-project/logical_reasoning/data/input/validation_split_5000.jsonl'
    output_file = '/data/disk4/home/chenrui/ai-project/logical_reasoning/data/input/validation_inst_complex_5000.json'
    tmp_file = '/data/disk4/home/chenrui/ai-project/logical_reasoning/data/input/tmp/tmp.jsonl'
    tmp_dir = '/data/disk4/home/chenrui/ai-project/logical_reasoning/data/input/tmp'

    need_split = False  # 是否需要切分子问题
    task_type = 'complex'  # 任务类型：raw 代表原版提示词
    generate_instruction_dataset(input_file, output_file, tmp_file, task_type, need_split)

    # # 清除生成的临时文件
    # if os.path.exists(tmp_dir):
    #     shutil.rmtree(tmp_dir)
    #     os.makedirs(tmp_dir)
    #     print(f'The folder {tmp_dir} has been cleared.')
    # else:
    #     print(f'The folder {tmp_dir} does not exist.')