import json
import pandas as pd

# 假设文件名为'questions.jsonl'
file_path = '/data/disk4/home/chenrui/ai-project/logical_reasoning/data/input/raw/round1_test_data.jsonl'
output_file = '/data/disk4/home/chenrui/ai-project/logical_reasoning/data/output/excel/round1_test_data.xlsx'

# 用于存储数据的列表
data = []

# 打开并读取文件
with open(file_path, 'r', encoding='utf-8') as file:
    for line in file:
        # 解析每一行的JSON
        record = json.loads(line.strip())
        
        # 提取ID
        record_id = record['id']
        
        # 提取问题描述
        problem = record['problem']
        
        # 提取选择题
        questions = record['questions']
        for question in questions:
            question_text = question['question']
            options = question['options']
            options = '\n'.join(f"{'ABCDEFG'[i]}. {o}" for i, o in enumerate(options))
            
            # 将数据添加到列表中
            data.append([record_id, problem, question_text, options])

# 创建DataFrame
df = pd.DataFrame(data, columns=['ID', 'Problem', 'Question', 'Options'])

# 将DataFrame写入xlsx文件
df.to_excel(output_file, index=False)

print(f"数据已成功写入 {output_file}")

