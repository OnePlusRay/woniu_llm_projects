import json
import random

input_file = '/data/disk4/home/chenrui/ai-project/logical_reasoning/LLaMA-Factory/data/logical_problems_large.json'
output_file = '/data/disk4/home/chenrui/ai-project/logical_reasoning/LLaMA-Factory/data/logical_problems_10000.json'

# 读取文件并加载数据
with open(input_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 随机筛选5000条数据
sample_size = 10000
if len(data) > sample_size:
    sampled_data = random.sample(data, sample_size)
    remaining_data = [item for item in data if item not in sampled_data]
else:
    print(f"数据不足 {sample_size} 条，返回所有数据。")
    sampled_data = data
    remaining_data = []

# 将选中的5000条数据写入新文件
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(sampled_data, f, ensure_ascii=False, indent=2)

# 将剩余的数据写回原始文件
with open(input_file, 'w', encoding='utf-8') as f:
    json.dump(remaining_data, f, ensure_ascii=False, indent=2)

print(f"已随机提取 {len(sampled_data)} 条数据写入 {output_file}，并从 {input_file} 中删除。")
