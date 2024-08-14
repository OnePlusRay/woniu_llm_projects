import os
import re
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from data_processing.utils import get_prompt, get_prompt_complex
from data_processing.compare_answer import load_jsonl, compare_results
from collections import Counter
import datetime
from tqdm import tqdm
import gc
from src.rag import EmbeddingModel, VectorStoreIndexBatch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 指定使用的设备

# 加载 embedding 模型
embed_model_path = 'models/bge-small-zh-v1.5'  # 还有 large 版本，可以考虑尝试   
embed_model = EmbeddingModel(embed_model_path)  # 加载 embedding 模型

# 加载向量知识库
doecment_path = 'data/external_data/knowledge_20000.txt'
index = VectorStoreIndexBatch(doecment_path, embed_model)
k = 5

# 释放多余的预留内存
gc.collect()
torch.cuda.empty_cache()

# 载模型和LoRA权重
model_path = 'models/internlm2_5-20b-chat'
lora_path = 'LLaMA-Factory/saves/internlm2.5-chat-20000-mini/lora/sft/checkpoint-500-best'  # 这里改称你的 lora 输出对应 checkpoint 地址

# 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# 加载模型
model = AutoModelForCausalLM.from_pretrained(model_path, device_map='auto', torch_dtype=torch.bfloat16, trust_remote_code=True).eval()

print(f"Model is on device: {next(model.parameters()).device}")

# 加载lora权重
model = PeftModel.from_pretrained(model, model_id=lora_path)

# 读取JSONL文件
data = []
input_file = '/data/disk4/home/chenrui/ai-project/logical_reasoning/data/input/raw/round1_test_data.jsonl'
with open(input_file, 'r', encoding='utf-8') as f:
    data = [json.loads(line.strip()) for line in f if line.strip()]

# 批量推理并保存结果
def extract_and_replace_subpath(path):
    saves_index = path.find('/saves/')
    if saves_index != -1:
        subpath = path[saves_index + len('/saves/'):]
        subpath_replaced = subpath.replace('/', '_')
        return subpath_replaced
    else:
        return None
    
output_file = f'/data/disk4/home/chenrui/ai-project/logical_reasoning/data/output/submit/submit_{os.path.basename(model_path)}_{extract_and_replace_subpath(lora_path)}_{k}_shot_rag.jsonl'

# 如果输出文件已经存在，则创建一个新的文件名
if os.path.exists(output_file):
    base_name, ext = os.path.splitext(output_file)
    counter = 1
    while True:
        new_output_file = f"{base_name}_{counter}{ext}"
        if not os.path.exists(new_output_file):
            output_file = new_output_file
            break
        counter += 1

# 若输出路径文件不存在，则创建文件
try:
    with open(output_file, 'w', encoding='utf-8') as f:
        pass  # 创建文件
except OSError as e:
    raise RuntimeError(f"Failed to create output file {output_file}: {e}")

# # 清空输出文件内容
# if os.path.exists(output_file):
#     with open(output_file, 'w', encoding='utf-8') as f:
#         f.write('')

results = []
total_questions = 0
batch_size = 50  # 每处理50个problem写入文件一次
tqdm1 = tqdm  # 进度条

for idx, item in enumerate(tqdm1(data)):
    problem = item["problem"]
    questions = item['questions']
    item_id = item["id"]
    item_results = {"id": item_id, "questions": []}

    for question in questions:
        question_text = question['question']
        options = question["options"]
        
        prompt = get_prompt(problem, question_text, options)

        context = index.query(prompt, k)

        if context:
            prompt = f'背景：{context}\n\n问题：{prompt}\n请基于背景，给出答案。'
        # print(prompt)
        messages = [
            {"role": "user", "content": prompt}
        ]
        inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_tensors="pt", return_dict=True).to('cuda')

        gen_kwargs = {"max_length": 2500, "do_sample": True, "top_k": 1}
        # 存储多次调用的输出
        outputs_list = []
        
        # 三次调用模型
        for i in range(3):
            with torch.no_grad():
                outputs = model.generate(**inputs, **gen_kwargs)
                outputs = outputs[:, inputs['input_ids'].shape[1]:]
                output = tokenizer.decode(outputs[0], skip_special_tokens=True)
                match_1 = re.match(r'([A-G])\.', output)
                match_2 = re.findall(r'[A-G]', output)
                if match_1:
                    output = match_1.group(1)
                elif match_2:
                    output = match_2[-1]
                outputs_list.append(output)
                # print(f"Model output {i}: {output}")

        # 进行多路投票
        vote_counts = Counter(outputs_list)
        final_output = vote_counts.most_common(1)[0][0]  # 选择出现次数最多的结果
        # print(f"-----> Final voted output: {final_output}")

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
        # for diff in differences:
        #     print(f"ID: {diff['id']}, Question Index: {diff['question_index']}, Answer in our results: {diff['answer_file1']}, Answer in Qwen2-72B: {diff['answer_file2']}")

        different_questions_count = len(differences)

        # 计算相同问题的数量
        same_questions_count = total_questions - different_questions_count

        # 计算相似度比例
        similarity_ratio = same_questions_count / total_questions if total_questions > 0 else 0

        print('\n' + f"Processed {idx + 1} problems, results written to {output_file}")
        print(f"Processed {total_questions} questions, different questions: {different_questions_count}")
        # print(f"Different Questions: {different_questions_count}")
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
# for diff in differences:
#     print(f"ID: {diff['id']}, Question Index: {diff['question_index']}, Answer in our results: {diff['answer_file1']}, Answer in Qwen2-72B: {diff['answer_file2']}")

# total_questions = 1328
different_questions_count = len(differences)

# 计算相同问题的数量
same_questions_count = total_questions - different_questions_count

# 计算相似度比例
similarity_ratio = same_questions_count / total_questions if total_questions > 0 else 0

print(f"Different Questions: {different_questions_count}")
print(f"Final Similarity Ratio: {similarity_ratio:.2%}")
print(f"Results written to {output_file}")

# 输出使用的模型信息和最终正确率
print(f"Model: {model_path}")
print(f"Lora adapter: {lora_path}")

# 重命名输出文件以包含正确率
new_output_file = f"{os.path.splitext(output_file)[0]}_sim_{similarity_ratio:.2%}{os.path.splitext(output_file)[1]}"
os.rename(output_file, new_output_file)
print(f"Renamed output file to {new_output_file}")