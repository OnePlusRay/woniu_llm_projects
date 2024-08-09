import os
import re
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from data_processing.utils import get_prompt
from data_processing.compare_answer import load_jsonl, compare_results
from collections import Counter

# os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # 只使用第一张显卡
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 指定使用的设备
ID = 'gemma-27b'

# 加载两个JSONL文件
output_file = f'/data/disk4/home/chenrui/ai-project/logical_reasoning/data/output/submit/submit_{ID}.jsonl'
baseline_file_path = '/data/disk4/home/chenrui/ai-project/logical_reasoning/data/output/test/Qwen2-72B-test.jsonl'

# 加载模型和LoRA权重
model_path = '/data/disk4/home/chenrui/gemma-2-27b-it-bnb-4bit'

# 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# 加载模型
model = AutoModelForCausalLM.from_pretrained(model_path, device_map={"": "cuda:2"}, torch_dtype=torch.bfloat16, trust_remote_code=True).eval()

print(f"Model is on device: {next(model.parameters()).device}")

# 读取JSONL文件
data = []
input_file = '/data/disk4/home/chenrui/ai-project/logical_reasoning/data/input/raw/round1_test_data.jsonl'
with open(input_file, 'r', encoding='utf-8') as f:
    data = [json.loads(line.strip()) for line in f if line.strip()]

# 批量推理并保存结果
output_file = f'/data/disk4/home/chenrui/ai-project/logical_reasoning/data/output/submit/submit_{ID}.jsonl'

# 清空输出文件内容
if os.path.exists(output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('')

results = []
total_questions = 0
batch_size = 50  # 每处理50个problem写入文件一次

for idx, item in enumerate(data):
    problem = item["problem"]
    questions = item['questions']
    item_id = item["id"]
    item_results = {"id": item_id, "questions": []}

    for question in questions:
        question_text = question['question']
        options = question["options"]

        prompt = get_prompt(problem, question_text, options)
        messages = [
            {"role": "user", "content": prompt}
        ]
        inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_tensors="pt", return_dict=True).to('cuda:2')

        gen_kwargs = {"max_length": 2500, "do_sample": True, "top_k": 1}
        # 存储多次调用的输出
        outputs_list = []
        
        # 三次调用模型
        for i in range(3):
            with torch.no_grad():
                outputs = model.generate(**inputs, **gen_kwargs)
                outputs = outputs[:, inputs['input_ids'].shape[1]:]
                output = tokenizer.decode(outputs[0], skip_special_tokens=True)
                print(f"Raw output: {output}")
                match_1 = re.match(r'([A-G])\.', output)
                match_2 = re.findall(r'[A-G]', output)
                if match_1:
                    output = match_1.group(1)
                elif match_2:
                    output = match_2[-1]
                outputs_list.append(output)
                print(f"Model output {i}: {output}")

        # 进行多路投票
        vote_counts = Counter(outputs_list)
        final_output = vote_counts.most_common(1)[0][0]  # 选择出现次数最多的结果
        print(f"-----> Final voted output: {final_output}")

        # 保存答案
        item_results["questions"].append({"answer": final_output})
        total_questions += 1

    results.append(item_results)

    # 每处理 batch_size 个problem写入文件一次，并打印提示
    if (idx + 1) % batch_size == 0:
        with open(output_file, 'a', encoding='utf-8') as f:  # 使用追加模式
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')

        # 比较输出结果和用 Qwen2-72B 生成的结果
        answer = load_jsonl(output_file)
        baseline_answer = load_jsonl(baseline_file_path)

        # 比较结果
        differences = compare_results(answer, baseline_answer)

        # 输出不同的结果
        for diff in differences:
            print(f"ID: {diff['id']}, Question Index: {diff['question_index']}, Answer in our results: {diff['answer_file1']}, Answer in Qwen2-72B: {diff['answer_file2']}")

        different_questions_count = len(differences)

        # 计算相同问题的数量
        same_questions_count = total_questions - different_questions_count

        # 计算相似度比例
        similarity_ratio = same_questions_count / total_questions if total_questions > 0 else 0

        print(f"Processed {idx + 1} problems, results written to {output_file}")
        print(f"Processed {total_questions} questions")
        print(f"Different Questions: {different_questions_count}")
        print(f"Similarity Ratio: {similarity_ratio:.2%}")

        results = []  # 清空结果列表以便存储下一批结果

# 写入剩余的结果
if results:
    with open(output_file, 'a', encoding='utf-8') as f:  # 使用追加模式
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    print(f"Processed {idx + 1} problems, results written to {output_file}")

# 比较输出结果和用 Qwen2-72B 生成的结果
answer = load_jsonl(output_file)
baseline_answer = load_jsonl(baseline_file_path)

# 比较结果
differences = compare_results(answer, baseline_answer)

# 输出不同的结果
for diff in differences:
    print(f"ID: {diff['id']}, Question Index: {diff['question_index']}, Answer in our results: {diff['answer_file1']}, Answer in Qwen2-72B: {diff['answer_file2']}")

# total_questions = 1328
different_questions_count = len(differences)

# 计算相同问题的数量
same_questions_count = total_questions - different_questions_count

# 计算相似度比例
similarity_ratio = same_questions_count / total_questions if total_questions > 0 else 0

print(f"Different Questions: {different_questions_count}")
print(f"Similarity Ratio: {similarity_ratio:.2%}")
