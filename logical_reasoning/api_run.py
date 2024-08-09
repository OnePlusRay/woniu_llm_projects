# from src.get_ans import get_ans
from src.get_ans_con import get_ans
import time

start_time = time.time()

if __name__ == '__main__':
    input_file = "/data/disk4/home/chenrui/ai-project/logical_reasoning/data/input/gpt/new_data_gpt4_25299.jsonl"
    output_file = "/data/disk4/home/chenrui/ai-project/logical_reasoning/data/input/gpt/gpt4_25299_checkcheck.jsonl"
    is_train = True  # 是否使用训练集
    
    get_ans(input_file, output_file, is_train)

    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total running time: {total_time} seconds")


# 训练集前 10 样本测试
# gpt4: 21/27
# qwen2_72b: 23/27

# 训练集前 20 样本测试
# gpt4: 41/52
# qwen2_72b: 42/51

# 训练集前 50 样本测试]
# gpt4: 0.84