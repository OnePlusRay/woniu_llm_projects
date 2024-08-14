import json
import random
import pdb

# 读取JSON文件
input_file = '/data/disk4/home/chenrui/ai-project/logical_reasoning/data/input/raw/round1_train_val_data_instruction.json'  # 替换为你的输入文件路径
output_sample_file = '/data/disk4/home/chenrui/ai-project/logical_reasoning/data/input/raw/round1_validation_data_instruction.json'  # 替换为你的抽样输出文件路径
output_remaining_file = '/data/disk4/home/chenrui/ai-project/logical_reasoning/data/input/raw/round1_train_data_instruction.json'  # 替换为你的剩余数据输出文件路径

# 读取数据
with open(input_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 随机抽取300条数据
sample_size = 300
if len(data) < sample_size:
    print(f"数据条目不足，无法抽取{sample_size}条数据。")
else:
    sampled_data = random.sample(data, sample_size)

    # 从原数据中去掉抽取的数据
    remaining_data = [item for item in data if item not in sampled_data]

    # 保存抽取的数据到新的JSON文件
    with open(output_sample_file, 'w', encoding='utf-8') as f:
        json.dump(sampled_data, f, ensure_ascii=False, indent=4)

    # 保存剩余的数据到新的JSON文件
    with open(output_remaining_file, 'w', encoding='utf-8') as f:
        json.dump(remaining_data, f, ensure_ascii=False, indent=4)

    print(f"成功抽取{sample_size}条数据并保存到{output_sample_file}。")
    print(f"剩余数据已保存到{output_remaining_file}。")

