import os
import re
import json
import torch
import pdb
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from collections import Counter
import random
import time
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 指定使用的设备

# 加载模型和LoRA权重
model_path = '/data/disk4/home/chenrui/Qwen2-72B-Instruct-bnb-4bit'
lora_path = '/data/disk4/home/chenrui/ai-project/logical_reasoning/LLaMA-Factory/saves/qwen2-72b-large/lora/sft/checkpoint-200-best'  # 这里改称你的 lora 输出对应 checkpoint 地址

# 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# 加载模型
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True).eval()

# 加载lora权重
model = PeftModel.from_pretrained(model, model_id=lora_path)

# 读取JSONL文件（训练使用的指令集格式）
data = []
input_file = '/data/disk4/home/chenrui/ai-project/logical_reasoning/data/input/validation_inst_5000.json'  # 输入文件路径
with open(input_file, 'r', encoding='utf-8') as f:
    data = json.load(f)  # 读取 JSON 数据

# 设置随机种子
random_seed = 42
random.seed(random_seed)

# 随机筛选 200 条数据
sample_size = 1000
if len(data) >= sample_size:
    data = random.sample(data, sample_size)
else:
    print(f"数据不足 {sample_size} 条，返回所有数据。")

# 批量推理并计算正确率
correct_count = 0
total_count = 0
tqdm1 = tqdm

for idx, item in enumerate(tqdm1(data)):
    instruction = item['instruction']
    input_text = item['input']
    expected_output = item['output']

    prompt = instruction + input_text
    messages = [
        {"role": "user", "content": prompt}
    ]
    inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_tensors="pt", return_dict=True).to(device)

    gen_kwargs = {"max_length": 2500, "do_sample": True, "top_k": 1}
    
    # 存储多次调用的输出
    outputs_list = []
    
    # 三次调用模型
    for i in range(3):
        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)
            outputs = outputs[:, inputs['input_ids'].shape[1]:]
            output = tokenizer.decode(outputs[0], skip_special_tokens=True)
            # print(f"Raw output {i}: {output}")
            match = re.findall(r'[A-G]', output)
            if match:
                output = match[-1]
            outputs_list.append(output)
            # print(f"Processed output {i}: {output}")

    # 进行多次投票
    vote_counts = Counter(outputs_list)
    final_output = vote_counts.most_common(1)[0][0]  # 选择出现次数最多的结果
    # print(f"Final voted output: {final_output}, Expected output: {expected_output}")

    # 比较预测答案和正确答案
    if final_output == expected_output:
        correct_count += 1
    total_count += 1

    # 每处理 batch_size 条数据输出一次实时的正确率
    batch_size = sample_size // 10
    if (total_count) % batch_size == 0:
        accuracy = correct_count / total_count
        print(f"Processed {total_count} items, Current Accuracy: {accuracy:.2%}")


# 输出使用的模型信息和最终正确率
print(f"Model: {model_path}")
print(f"Lora adapter: {lora_path}")
print(f"Validation dataset: {input_file}")

accuracy = correct_count / total_count
print(f"Final Accuracy: {accuracy:.2%}")



