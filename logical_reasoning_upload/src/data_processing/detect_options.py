import json

# 输入和输出文件路径
input_file = '/data/disk4/home/chenrui/ai-project/logical_reasoning/data/input/external_data/external_data_25000.jsonl'
valid_output_file = '/data/disk4/home/chenrui/ai-project/logical_reasoning/data/input/external_data/external_data_valid.jsonl'
invalid_output_file = '/data/disk4/home/chenrui/ai-project/logical_reasoning/data/input/external_data/external_data_invalid.jsonl'
error_log_file = '/data/disk4/home/chenrui/ai-project/logical_reasoning/data/input/external_data/error_log.txt'

# 定义有效的输出字符
valid_outputs = set("ABCDEFG")

# 打开输入文件和输出文件
with open(input_file, 'r', encoding='utf-8') as infile, \
     open(valid_output_file, 'w', encoding='utf-8') as valid_outfile, \
     open(invalid_output_file, 'w', encoding='utf-8') as invalid_outfile, \
     open(error_log_file, 'w', encoding='utf-8') as error_log:
    
    # 逐行读取文件并处理JSON数据
    for line_number, line in enumerate(infile, start=1):
        try:
            item = json.loads(line.strip())
            valid = True
            for question in item['questions']:
                if question['answer'] not in valid_outputs:
                    valid = False
                    break
            
            if valid:
                valid_outfile.write(json.dumps(item, ensure_ascii=False) + '\n')
            else:
                invalid_outfile.write(json.dumps(item, ensure_ascii=False) + '\n')
        except Exception as e:
            error_log.write(f"Error processing line {line_number}: {line.strip()}\n")
            error_log.write(f"Exception: {e}\n")

print(f"Valid output data saved to {valid_output_file}")
print(f"Invalid output data saved to {invalid_output_file}")
print(f"Error log saved to {error_log_file}")
