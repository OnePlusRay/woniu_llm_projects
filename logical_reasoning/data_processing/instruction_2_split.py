import json

def convert_data(original_data_list):
    converted_data_list = []
    
    for index, original_data in enumerate(original_data_list):
        # 去掉 "题目\n" 并提取问题描述
        problem = original_data["input"].replace("题目:\n", "").strip()
        question = problem.split("问题:\n")[-1].strip().split("\n\n")[0]  # 提取问题部分
        
        # 提取选项
        options = []
        option_lines = original_data["input"].split("\n")[original_data["input"].split("\n").index("问题:") + 1:]  # 获取问题后的所有行
        for line in option_lines:
            if line.startswith("A.") or line.startswith("B.") or line.startswith("C.") or line.startswith("D."):
                options.append(line.split(". ")[1].strip())  # 提取选项内容
        
        answer = original_data["output"]
        id_value = f"round1_train_data_{index:03d}"  # 生成 ID，格式为 round1_train_data_000
        
        # 创建新的字典
        converted_data = {
            "problem": problem,
            "question": question,
            "options": options,
            "answer": answer,
            "id": id_value
        }
        
        converted_data_list.append(converted_data)
    
    return converted_data_list

# 示例输入数据
original_data_list = [
    {
        "instruction": "你是一个逻辑推理专家，擅长解决逻辑推理问题。以下是一个逻辑推理的题目，形式为单项选择题。所有的问题都是（close-world assumption）闭世界假设，即未观测事实都为假。请逐步分析问题，最终只输出选项，如'A'。",
        "input": "\n题目:\n有一个英文到法文的词汇表，包含以下对应词汇：\n\n1. the -> le\n2. cat -> chat\n3. jumps -> sauts\n4. over -> sur\n5. moon -> lune\n6. cow -> vache\n7. plays -> jouer\n8. fiddle -> violon\n9. egg -> bougre\n10. falls -> des chutes\n11. off -> de\n12. wall -> mur\n\n根据这个词汇表，翻译以下英文句子成法文：\n\n问题:\n选择题 1：\n英文句子 \"the cat jumps over the moon\" 翻译成法文是：\nA. le chat saute sur la lune\nB. le chat sauts sur le lune\nC. le sauts chat sur le lune\nD. le chat sauts sur le lune\n    ",
        "output": "D"
    },
    {
        "instruction": "你是一个逻辑推理专家，擅长解决逻辑推理问题。以下是一个逻辑推理的题目，形式为单项选择题。所有的问题都是（close-world assumption）闭世界假设，即未观测事实都为假。请逐步分析问题，最终只输出选项，如'A'。",
        "input": "\n题目:\n有一个英文到法文的词汇表，包含以下对应词汇：\n\n1. the -> le\n2. cat -> chat\n3. jumps -> sauts\n4. over -> sur\n5. moon -> lune\n6. cow -> vache\n7. plays -> jouer\n8. fiddle -> violon\n9. egg -> bougre\n10. falls -> des chutes\n11. off -> de\n12. wall -> mur\n\n根据这个词汇表，翻译以下英文句子成法文：\n\n问题:\n选择题 2：\n英文句子 \"the cow plays the fiddle\" 翻译成法文是：\nA. le vache jouer le violon\nB. le jouer vache le violon\nC. le vache jouer la vièle\nD. la vache joue le violon\n    ",
        "output": "A"
    }
]

# 调用函数进行转换
converted_data_list = convert_data(original_data_list)

# 保存为 JSONL 格式
jsonl_file_path = '/data/disk4/home/chenrui/ai-project/logical_reasoning/data/input/tmp/converted_data.jsonl'  # 输出文件路径
with open(jsonl_file_path, 'w', encoding='utf-8') as f:
    for item in converted_data_list:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')  # 每个 JSON 对象后加换行符

print(f"Data has been converted to JSONL format and saved to {jsonl_file_path}")
