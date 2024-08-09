import pdb
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 1. 加载本地模型
model_path = '/data/disk4/home/chenrui/ai-project/Qwen2-7B-Instruct'
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# 2. 定义提示词构建函数
def get_prompt(problem, question, options):
    '''构建一个逻辑推理问题的提示文本'''
    options = '\n'.join(f"{'ABCDEFG'[i]}. {o}" for i, o in enumerate(options))
    prompt = f"""
你是一个逻辑推理专家，擅长解决逻辑推理问题。以下是一个逻辑推理的题目，形式为单项选择题。所有的问题都是（close-world assumption）闭世界假设，即未观测事实都为假。请逐步分析问题并在最后一行输出答案，最后一行的格式为"答案是：A"。题目如下：

### 题目:
{problem}

### 问题:
{question}
{options}
    """
    return prompt

def extract_analysis_and_result(generated_text, prompt):
    """
    从生成的文本中提取分析过程和结果。

    参数:
    - generated_text: 生成的文本
    - prompt: 提示词

    返回:
    - 提取的分析过程和结果
    """
    # 去除提示词部分
    analysis_and_result = generated_text.replace(prompt, "").strip()
    return analysis_and_result

# 3. 定义推理函数
def generate_answer(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    outputs = model.generate(
        inputs,
        max_length=1024,  # 增加最大长度
        min_length=50,   # 设置最小长度
        num_beams=5,     # 使用束搜索
        no_repeat_ngram_size=2,  # 避免重复的 n-gram
        early_stopping=True,
        eos_token_id=tokenizer.eos_token_id,
        top_k=50,        # 使用 top-k 采样
        top_p=0.95       # 使用 top-p 采样
    )
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # pdb.set_trace()
    return answer

# 4. 示例数据
data = {"problem": "在一个社区中下面的居民特点已知：\n\n1. Lakeram是玩家。\n2. Efaan是黑客。\n3. David和Wakeel都喜欢动漫。\n4. Hemrant是流氓。\n\n任何被分类为黑客的人也被认为是酷的。\n\nAcquila讨厌所有酷的人和所有的玩家。  \nKheeram讨厌所有酷的人和喜欢动漫的人，也讨厌所有的流氓。\n\n此外，以下居民曾经在同一所学校上过学：\n- Sarid和Aaliya\n- Aaliya和Rudeshwar\n- Rudeshwar和Christopher\n\n同校的关系是可以传递的，如果A和B曾经同校，B和C也同校，那么A和C亦然。\n\n根据以上信息，回答以下选择题。", "question": "选择题 1：\n谁讨厌Efaan？", "options": ["Acquila", "Kheeram", "David", "Wakeel"], "answer": "A", "id": "round1_train_data_008"}
problem = data["problem"]
question = data['question']
options = data["options"]
prompt = get_prompt(problem, question, options)

# 5. 生成答案
device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
model.to(device)
answer = generate_answer(prompt)
answer_without_prompt = extract_analysis_and_result(answer, prompt)
print(answer_without_prompt)

