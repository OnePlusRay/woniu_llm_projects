import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType
import bitsandbytes as bnb

# 加载qwen1.5-32b只需要单卡，在终端中输入下面命令
# CUDA_VISIBLE_DEVICES=0 python quantize.py

# 加载qwen2-72b需要多卡，在终端中输入下面命令
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python quantize.py

# 加载分词器和模型
model_path = '/data/disk4/home/chenrui/.cache/modelscope/hub/Shanghai_AI_Laboratory/internlm2_5-20b-chat'
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)

# 配置8bit量化
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_8bit_compute_dtype=torch.bfloat16,
    bnb_8bit_use_double_quant=True,
    # bnb_8bit_quant_type='nf4'
)

# 加载8bit量化的模型
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map={'':'cuda'},
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    quantization_config=bnb_config
)

# 保存量化后的模型
save_path = '/data/disk4/home/chenrui/InternLM2_5-20B-Chat'
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

print(f"bnb-in8-quantize successfully, model is saved to {save_path}")