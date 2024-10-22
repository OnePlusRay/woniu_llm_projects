import os
import re
import gc
import json
import torch
import datetime
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from collections import Counter
from tqdm import tqdm

from src.rag import EmbeddingModel, VectorStoreIndexBatch
from data_processing.utils import get_prompt, get_prompt_complex
from data_processing.compare_answer import load_jsonl, compare_results

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 指定使用的设备

def print_gpu_info():
    '''打印GPU信息'''
    if torch.cuda.is_available():
        print("CUDA is available on this system.")
        print(f"Number of GPUs available: {torch.cuda.device_count()}")
        print("-" * 50)
        
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  Total Memory: {torch.cuda.get_device_properties(i).total_memory / (1024**2):.2f} MB")
            print(f"  Current device: {torch.cuda.current_device() == i}")
            print("-" * 50)

    else:
        print("CUDA is not available.")

print_gpu_info()

# 加载 embedding 模型
embed_model_path = '/home/user/data2/chenrui/models/AI-ModelScope/bge-small-zh-v1___5'  # 还有 large 版本，可以考虑尝试   
embed_model = EmbeddingModel(embed_model_path)  # 加载 embedding 模型

# 加载向量知识库
doecment_path = 'data/external_data/knowledge_20000.txt'
index = VectorStoreIndexBatch(doecment_path, embed_model)
k = 5

# 释放多余的预留内存
gc.collect()
torch.cuda.empty_cache()

# 载模型和LoRA权重
model_path = '/home/user/data2/chenrui/models/chenr1209/InternLM2___5-20B-Chat-bnb-int8'
lora_path = '/home/user/data2/chenrui/models/chenr1209/Intern2___5-20B-Chat-bnb-int8-lora_adapter'  # 这里改称你的 lora 输出对应 checkpoint 地址

# 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# 加载模型
model = AutoModelForCausalLM.from_pretrained(model_path, device_map='auto', torch_dtype=torch.bfloat16, trust_remote_code=True).eval()

# 加载lora权重
model = PeftModel.from_pretrained(model, model_id=lora_path)

# 使用 DataParallel
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))

# 读取JSONL文件
data = []
input_file = 'data/input/raw/round1_test_data.jsonl'
with open(input_file, 'r', encoding='utf-8') as f:
    data = [json.loads(line.strip()) for line in f if line.strip()]

output_file = 'data/output/submit/submit_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + '.jsonl'

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
        inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_tensors="pt", return_dict=True).to(device)

        gen_kwargs = {"max_length": 4096, "do_sample": True, "top_k": 1}
        # 存储多次调用的输出
        outputs_list = []
        
        # 三次调用模型
        for i in range(3):
            with torch.no_grad():
                outputs = model.module.generate(**inputs, **gen_kwargs)
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

        results = []  # 清空结果列表以便存储下一批结果

# 写入剩余的结果
if results:
    with open(output_file, 'a', encoding='utf-8') as f:  # 使用追加模式
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    print(f"Processed {idx + 1} problems, results written to {output_file}")




