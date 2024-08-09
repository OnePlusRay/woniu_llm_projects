import json

def load_jsonl(file_path):
    """加载JSONL文件并返回一个列表"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line.strip()))
    return data

def compare_results(data1, data2):
    """比较两个数据集并找出不同的结果"""
    differences = []
    for item1, item2 in zip(data1, data2):
        if item1["id"] != item2["id"]:
            differences.append((item1, item2))
        else:
            for idx, (q1, q2) in enumerate(zip(item1["questions"], item2["questions"])):
                if q1["answer"] != q2["answer"]:
                    differences.append({
                        "id": item1["id"],
                        "question_index": idx,
                        "answer_file1": q1["answer"],
                        "answer_file2": q2["answer"]
                    })
    return differences

if __name__ == '__main__':
    # 加载两个JSONL文件
    file_path1 = '/data/disk4/home/chenrui/ai-project/logical_reasoning/data/output/submit/submit_4.jsonl'
    file_path2 = '/data/disk4/home/chenrui/ai-project/logical_reasoning/data/output/submit/submit_5.jsonl'

    data1 = load_jsonl(file_path1)
    data2 = load_jsonl(file_path2)

    # 比较结果
    differences = compare_results(data1, data2)

    # 输出不同的结果
    for diff in differences:
        print(f"ID: {diff['id']}, Question Index: {diff['question_index']}, Answer in File 1: {diff['answer_file1']}, Answer in File 2: {diff['answer_file2']}")

    # 输出不一样问题的个数
    print(f"Number of different questions: {len(differences)}")