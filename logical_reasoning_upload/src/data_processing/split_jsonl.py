import json
import pdb

def split_dataset(input_file, output_file):
    """
    将数据集中的每个子问题作为一个独立的问题，拆分数据集并保存到新的文件中。

    参数:
    - input_file: 输入文件路径
    - output_file: 输出文件路径
    """
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        new_id = 1000
        for line_number, line in enumerate(infile, start=1):
            line = line.strip()
            if not line:  # 跳过空行
                continue
            try:
                item = json.loads(line)
                problem = item['problem']
                for question in item['questions']:
                    new_item = {
                        'problem': problem,
                        'question': question['question'],
                        'options': question['options'],
                        'answer': question.get('answer', None),  # 如果有答案则保留
                        'Qwen2-72B':question.get('Qwen2-72B', None),
                        'id': f'round1_train_data_{new_id}'
                    }
                    outfile.write(json.dumps(new_item, ensure_ascii=False) + '\n')
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON on line {line_number}: {line}")
                print(e)
            new_id += 1
    return output_file

# 输入文件路径
input_file = '/data/disk4/home/chenrui/ai-project/logical_reasoning/data/input/gpt/new_data_gpt4_25288_1.jsonl'
# 输出文件路径
output_file = '/data/disk4/home/chenrui/ai-project/logical_reasoning/data/input/gpt/gpt4_4590_806.jsonl'

# 拆分数据集
split_dataset(input_file, output_file)

