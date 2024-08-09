import os
import re
import pdb
import json
import torch
import optuna
from datasets import Dataset
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer, GenerationConfig, EarlyStoppingCallback, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType
from sklearn.model_selection import train_test_split
import bitsandbytes as bnb

# 加载qwen1.5-32b只需要单卡，在终端中输入下面命令
# CUDA_VISIBLE_DEVICES=0 python quantize.py

# 加载qwen2-72b需要多卡，在终端中输入下面命令
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python quantize.py

# 加载分词器和模型
model_path = '/data/disk4/home/chenrui/.cache/modelscope/hub/01ai/Yi-1___5-34B-Chat'
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)

# 配置4bit量化
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type='nf4'
)

# 加载4bit量化的模型
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map={'':'cuda'},
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    quantization_config=bnb_config
)

# 保存量化后的模型
save_path = '/data/disk4/home/chenrui/Yi-1.5-34B-Chat-bnb-4bit'
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

print(f"bnb-in4-quantize successfully, model is saved to {save_path}")
