# 指定多卡训练

import os
import re
import pdb
import json
import torch
import optuna
from datasets import Dataset
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer, GenerationConfig, EarlyStoppingCallback
from peft import get_peft_model, LoraConfig, TaskType
from sklearn.model_selection import train_test_split

GPU_ID = 2
os.environ["CUDA_VISIBLE_DEVICES"] = f"{GPU_ID}"  # 只使用第一张显卡
device = torch.device(f"cuda:{GPU_ID}" if torch.cuda.is_available() else "cpu")  # 指定使用的设备
device_map = {"": f"cuda:{GPU_ID}"}
TASK_ID = 2
# torch.cuda.device_count()
# # 指定使用第1号GPU
# torch.cuda.set_device(0)
# # 检查设置是否生效
# print(torch.cuda.current_device())  # 应该输出1
# print(torch.cuda.get_device_name(0))  # 输出第1号GPU的名称

with torch.device(device):
    # # 读取原数据集
    # df1 = pd.read_json('/data/disk4/home/chenrui/ai-project/logical_reasoning/data/input/raw/round1_train_data_instruction.json')

    # 读取新增数据集
    df2 = pd.read_json('/data/disk4/home/chenrui/ai-project/logical_reasoning/data/input/new/train_data_new_inst.jsonl')  # 人工减弱噪声后的数据集

    # # 读取生成数据集
    df3 = pd.read_json('/data/disk4/home/chenrui/ai-project/logical_reasoning/data/input/gpt/gpt_802_inst.json')

    # 合并两个 DataFrame
    merged_df = pd.concat([df2, df3], ignore_index=True)
    print(len(merged_df))

    # # 合并好的指令集
    # merged_df = pd.read_json('/data/disk4/home/chenrui/ai-project/logical_reasoning/data/input/all_data_inst_0729.json')

    # 划分训练集和验证集
    train_data, val_data = train_test_split(merged_df, test_size=0.2, random_state=42)  # 80% 训练，20% 验证
    # print(val_data)

    # 将验证集保存为JSON格式到tmp文件夹
    tmp_dir = '/data/disk4/home/chenrui/ai-project/logical_reasoning/data/input/tmp'
    os.makedirs(tmp_dir, exist_ok=True)  # 创建tmp文件夹（如果不存在的话）
    val_data.to_json(os.path.join(tmp_dir, 'validation_data.json'), orient='records', force_ascii=False)

    # 将数据集转换为Dataset对象
    train_ds = Dataset.from_pandas(train_data)
    val_ds = Dataset.from_pandas(val_data)

    train_ds

    # 加载分词器和模型
    model_path = '/data/disk4/home/chenrui/ai-project/Qwen2-7B-Instruct'
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map='auto', torch_dtype=torch.bfloat16)
    model.enable_input_require_grads() # 开启梯度检查点时，要执行该方法
    model.dtype # 查看精度
    print(next(model.parameters()).device)

    def process_func(example):
        MAX_LENGTH = 1800    # Llama分词器会将一个中文字切分为多个token，因此需要放开一些最大长度，保证数据的完整性
        input_ids, attention_mask, labels = [], [], []
        instruction = tokenizer(f"<|im_start|>system\n你是一个逻辑推理专家，擅长解决逻辑推理问题。<|im_end|>\n<|im_start|>user\n{example['instruction'] + example['input']}<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False)  # add_special_tokens 不在开头加 special_tokens
        response = tokenizer(f"{example['output']}", add_special_tokens=False)
        input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
        attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]  # 因为eos token咱们也是要关注的所以 补充为1
        labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]  
        if len(input_ids) > MAX_LENGTH:  # 做一个截断
            input_ids = input_ids[:MAX_LENGTH]
            attention_mask = attention_mask[:MAX_LENGTH]
            labels = labels[:MAX_LENGTH]
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }
    # 处理训练集和验证集
    train_ds = train_ds.map(process_func, remove_columns=train_ds.column_names)
    val_ds = val_ds.map(process_func, remove_columns=val_ds.column_names)
    num_train_samples = len(train_ds)
    print(train_ds, tokenizer.decode(train_ds[0]['input_ids']))
    print(tokenizer.decode(list(filter(lambda x: x != -100, train_ds[1]["labels"]))))


    # 设置 lora 参数
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, 
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        inference_mode=False, # 训练模式
        r=8, # Lora 秩
        lora_alpha=32,  # Lora alaph，具体作用参见 Lora 原理
        lora_dropout=0.2  # Dropout 比例
    )
    model = get_peft_model(model, config)

    # 查看可训练参数
    model.print_trainable_parameters()


    # 自定义 Trainer 类
    class CustomTrainer(Trainer):
        def training_step(self, model, inputs):
            loss = super().training_step(model, inputs)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 梯度裁剪
            return loss


    # 设置训练轮次和批量大小
    per_device_train_batch_size = 32
    num_train_epochs = 5
    gradient_accumulation_steps = 1

    # 设置训练超参数
    args = TrainingArguments(
        output_dir=f"./checkpoints/Qwen2_7B_instruct_lora_{TASK_ID}",
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=32,
        num_train_epochs=num_train_epochs,
        logging_steps=10,
        save_steps=50,
        learning_rate=3e-5,
        save_on_each_node=True,
        # gradient_checkpointing=True,
        lr_scheduler_type="linear",
        warmup_steps=100,
        gradient_accumulation_steps=gradient_accumulation_steps,
        evaluation_strategy="steps",  # 设置评估策略
        eval_steps=25,
        # fp16=True,  # 使用混合精度训练，防止梯度爆炸
        load_best_model_at_end=True,  # 在训练结束时加载最佳模型
        metric_for_best_model="eval_loss",  # 监控的指标
        greater_is_better=False,  # 因为我们希望损失最小化
        logging_dir='./logs',  # TensorBoard 日志目录
        report_to="tensorboard",  # 启用 TensorBoard
        # dataloader_num_workers=4,
        # distributed_data_parallel=True,
    )

    # 查看训练总步数
    total_training_steps = (num_train_samples // per_device_train_batch_size) * num_train_epochs
    print(f"Total training steps: {total_training_steps}")


    # 训练 可用 CustomTrainer
    trainer = CustomTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,  # 添加验证集
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],  # 设置早停法，耐心为 3
    )
    trainer.train()

    trainer.save_model(f"./output/Qwen2_7B_instruct_lora_{TASK_ID}_final")