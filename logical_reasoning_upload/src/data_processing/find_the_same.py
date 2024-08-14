import json
from utils import split_dataset

def process_jsonl_file(ifn):
    data = []
    # 读取 JSONL 文件并将数据转换为列表
    with open(ifn, 'r', encoding='utf-8') as reader:
        for line in reader:
            sample = json.loads(line)
            data.append(sample)
    return data

if __name__ == '__main__':
    ifn = '/data/disk4/home/chenrui/ai-project/logical_reasoning/data/input/gpt/gpt_8000_808.jsonl'  # 子问题拆分后的 jsonl 
    ofn = '/data/disk4/home/chenrui/ai-project/logical_reasoning/data/input/gpt/gpt_8000_808_split.jsonl'  # 输出文件路径
    tfn = '/data/disk4/home/chenrui/ai-project/logical_reasoning/data/input/tmp/gpt4.jsonl'

    split_dataset(ifn, tfn)
    data = process_jsonl_file(tfn)

    filtered_data = [item for item in data if item['answer'] == item['gpt4']]

    # 将筛选后的数据写入 jsonl 文件
    with open(ofn, 'w', encoding='utf-8') as f:
        for item in filtered_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print("筛选后的数据已写入 filtered_data.jsonl 文件。")