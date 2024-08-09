import json
import pdb

def read_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

def write_jsonl(file_path, data):
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def find_unique_lines(file1, file2, output_file):
    data1 = read_jsonl(file1)
    data2 = read_jsonl(file2)

    # 将 data2 转换为集合以便于查找，使用 json.dumps 进行序列化
    data2_set = {json.dumps(item, sort_keys=True) for item in data2}

    # 找到在 file1 中但不在 file2 中的行
    unique_to_file1 = [item for item in data1 if json.dumps(item, sort_keys=True) not in data2_set]

    # 写入新的 JSONL 文件，保留原始数据
    write_jsonl(output_file, unique_to_file1)


file1_path = '/data/disk4/home/chenrui/ai-project/logical_reasoning/data/input/new/train_data_new_2_split.jsonl'
file2_path = '/data/disk4/home/chenrui/ai-project/logical_reasoning/data/input/raw/round1_train_data_split.jsonl'
output_file1_path = '/data/disk4/home/chenrui/ai-project/logical_reasoning/data/output/only_new.jsonl'
output_file2_path = 'unique_to_file2.jsonl'

# 使用示例
find_unique_lines(file1_path, file2_path, output_file1_path)

