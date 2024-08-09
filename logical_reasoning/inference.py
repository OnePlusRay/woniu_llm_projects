import os
import re
import json
import torch
import pdb
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from collections import Counter
import random

os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # 只使用第一张显卡
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")  # 指定使用的设备

# 加载模型和LoRA权重
model_path = '/data/disk4/home/chenrui/Qwen1.5-32B-Chat-bnb-4bit'
lora_path = '/data/disk4/home/chenrui/LLaMA-Factory-main/saves/qwen1.5-32b/lora/sft/checkpoint-675'  # 这里改称你的 lora 输出对应 checkpoint 地址

# 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# 加载模型
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True).eval()

# 加载lora权重
model = PeftModel.from_pretrained(model, model_id=lora_path)

# # 读取知识库（JSONL格式）
# knowledge_base = {}
# knowledge_file = '/data/disk4/home/chenrui/ai-project/logical_reasoning/data/input/raw/round1_train_data.jsonl'  # 替换为你的知识库文件路径
# with open(knowledge_file, 'r', encoding='utf-8') as f:
#     for line in f:
#         if line.strip():  # 确保行不为空
#             entry = json.loads(line.strip())
#             question = entry.get("question")
#             knowledge = entry.get("knowledge")
#             if question and knowledge:
#                 knowledge_base[question] = knowledge  # 将问题作为键，知识作为值

# 读取JSONL文件
data = []
input_file = '/data/disk4/home/chenrui/LLaMA-Factory-main/data/logical_problems_large.json'
with open(input_file, 'r', encoding='utf-8') as f:
    data = json.load(f)  # 读取 JSON 数据

# 随机筛选 100 条数据
sample_size = 250
if len(data) > sample_size:
    data = random.sample(data, sample_size)
else:
    print(f"数据不足 {sample_size} 条，返回所有数据。")

# 批量推理并计算正确率
correct_count = 0
total_count = 0

for idx, item in enumerate(data):
    instruction = item['instruction']
    input_text = item['input']
    expected_output = item['output']
    # match = re.findall(r'[A-G]', output)
    # if match:
    #     expected_output = match[-1]

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
            print(output)
            match = re.findall(r'[A-G]', output)
            if match:
                output = match[-1]
            outputs_list.append(output)
            print(f"Model output {i}: {output}")

    # 进行多路投票
    vote_counts = Counter(outputs_list)
    final_output = vote_counts.most_common(1)[0][0]  # 选择出现次数最多的结果
    print(f"Final voted output: {final_output}, Expected output: {expected_output}")

    # 比较预测答案和正确答案
    if final_output == expected_output:
        correct_count += 1
    total_count += 1

    # 每处理 n 条数据输出一次实时的正确率
    if (total_count) % 50 == 0:
        accuracy = correct_count / total_count
        print(f"Processed {total_count} items, Current Accuracy: {accuracy:.2%}")

# 最终正确率
accuracy = correct_count / total_count
print(f"Final Accuracy: {accuracy:.2%}")





