import json

# 输入和输出文件路径
input_file = '/data/disk4/home/chenrui/ai-project/logical_reasoning/data/input/tmp/tmp_ok.jsonl'
output_file = '/data/disk4/home/chenrui/ai-project/logical_reasoning/data/input/gpt4.jsonl'

# 读取JSON文件
with open(input_file, 'r', encoding='utf-8') as infile:
    data = [json.loads(line.strip()) for line in infile]

# 创建一个字典，用于合并数据
merged_data = {}

# 遍历数据，合并相同 problem 和 id 的条目
for item in data:
    problem = item['problem']
    question = item['question']
    options = item['options']
    answer = item['answer']
    item_id = item['id']
    
    if item_id not in merged_data:
        merged_data[item_id] = {
            "problem": problem,
            "questions": [],
            "id": item_id
        }
    
    merged_data[item_id]["questions"].append({
        "question": question,
        "options": options,
        "answer": answer
    })

# 将合并后的数据写入输出文件
with open(output_file, 'w', encoding='utf-8') as outfile:
    for item_id, item in merged_data.items():
        outfile.write(json.dumps(item, ensure_ascii=False) + '\n')

print(f"Merged data saved to {output_file}")
