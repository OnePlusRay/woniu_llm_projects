# 大模型的复杂逻辑推理能力评估

用扩充后的给定数据集对大模型进行微调，并在推理过程中增加RAG等技术，使其在复杂逻辑推理问题中获得较好的表现。

## 整体流程
1. 调用 GPT4 的 API 接口扩充数据集（从 500 扩充到 10000 以上）
2. 数据清洗和处理
3. 对 InternLM2.5-20B-Chat 进行 bnb-int8 量化，然后 LoRA 指令监督微调，并用验证集进行正确率测试
4. 用微调得到的 LoRA 权重与原模型合并后进行推理，通过设计提示词模板，建立外部知识库检索增强生成（RAG）得到最后提交的结果

## 方法介绍
本项目的技术路线主要分为两个方面：
- 通过微调增强模型的逻辑推理能力
- 通过推理过程中的提示词工程和 RAG 等技术让模型发挥最大能力
### 训练
我们选择了书生·浦语2.5-20B-对话模型，进行 bnb-int8 量化，然后使用 LLaMA-Factory 框架进行 QLoRA 微调。
### 推理
模型推理的具体流程：
1. 当获取到待处理的问题后，首先通过一个 Embedding 模型把当前问题和知识库的内容向量化，计算当前问题和向量知识库中每个文档的相似度，返回 k 个相似度最高的文档
2. 返回的这 k 个文档作为提示词的背景部分，跟当前的问题一起作为完整的提示词，作为大模型的输入
3. 大模型接受输入的内容后进行处理，输出结果，经过一定的后处理得到本次处理的答案，每个问题需要处理 3 次，选择出现次数最多的答案

涉及的方法：
- 提示词过程
- RAG
- 多路投票


## 环境依赖
### 硬件环境
- 操作系统：Ubuntu 22.04.3 LTS
- GPU：NVIDIA A100 80GB PCIe
- CUDA Version: 12.2
- CUDNN version: 90100
### 软件环境
- Python：3.10.14
- Pytorch：2.4.0
- transformers：4.43.4
- peft：0.12.0
- datasets：2.20.0
- accelerate：0.32.0
- trl：0.9.6
- tqdm：4.66.5
- Bitsandbytes：0.43.3
### 使用模型
- 大语言模型：internlm2_5-20b-chat
- Embedding模型：bge-small-zh-v1.5


## 文件结构
### 代码部分
- src/：主函数
  - ```main.py```：入口程序运行的脚本，一键生成测试集答案
  - ```quantize.py```：模型量化
  - ```train_lora.ipynb```：LoRA 微调训练 + 验证集测试
  - ```api_run.py```：调用大模型 API 运行脚本
  - ```validation.py```：验证集测试
  - ```download.py```：下载模型
- src/src/：核心处理函数
  - ```get_ans.py```：回答逻辑推理问题
  - ```get_ans_con.py```：多线程并发版本（速度快）
  - ```generate_problems.py```：调用 GPT4 生成数据
  - ```rag.py```：RAG 检索增强生成
- src/data_processing/：数据处理相关函数
  - ```jsonl_2_instruction.py```：jsonl 格式转指令格式（可选是否需要拆分子问题）
  - ```split_jsonl.py```：对原数据集格式拆分子问题
  - ```find_the_same.py```：筛选出原问题答案和大模型生成答案一致的数据
  - ```inst_change_prompt.py```：更换指令集中的提示词
  - ```utils.py```：数据处理常用的辅助函数（生成提示词模板和提取答案）
- src/utils/：API 调用相关辅助函数（这些函数本地均无法使用）
  - ```chain.py```：通过 LangChain 构建 prompt-llm-output 调用链
  - ```llm.py```：可选的大模型 API 函数
- ```requirements.txt```：项目依赖
- ```run.sh```：用于一键推理生成测试集结果的入口程序

### 数据模型部分
- data/：输入和输出的数据（包括 jsonl 和 json 数据）
  - external_data/：生成的数据    
- submit/：提交结果

## 项目运行
若只需要推理生成测试集结果，我们提供了一个入口程序```run.sh```。

### 运行前安装
- 安装依赖（包括 LangChain、深度学习等相关库）
```bash
pip install -r requirements.txt
```
- 安装 LLaMA Factory（大模型微调和部署框架）
```bash
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]"
```
### 数据生成（大模型 API） 
这部分有两个功能，分别写在文件夹 src 中的不同 py 文件，包括：
- 生成数据（生成新问题）
- 回答逻辑推理问题

它们的操作流程几乎一致，统一在 ```api_run``` 导入后运行，若需要修改内部参数（如跑多少条数据写入一次文件，线程数，每个线程的任务数等）和代码逻辑，则需要在文件夹 src 中的 py 文件修改。

在我们的项目中，这一部分的流程如下：
1. 调用 GPT4 生成一批新数据，格式与原数据集一致（多线程）
2. 调用 claude 3.5 对生成的问题进行回答，并把答案存入原数据（多线程）

运行单个功能的流程如下：

1. 传入参数
```python
input_file = "/data/disk4/home/chenrui/ai-project/logical_reasoning/data/input/gpt/new_data_gpt4_25299.jsonl"  # 输入文件路径
output_file = "/data/disk4/home/chenrui/ai-project/logical_reasoning/data/input/gpt/gpt4_25299_checkcheck.jsonl"  # 输出文件路径
is_train = True  # 是否使用训练集（训练集有答案，可以做验证）
```

2. 终端运行
```bash
cd your_file_path
proxychains python main.py
```

### 数据处理
目前我们获得了一批跟原数据集格式相同的数据，包括两个大模型的答案（一个是生成时带有的，一个是根据题目回答的），可以简单地认为，如果一个问题能够让两个大模型回答一致（选项一致），则这个问题的质量可以接受。为了筛选出质量相对高的数据，我们简单地把两个答案一致的数据筛选出来，作为扩充的数据集。

操作步骤如下：
1. 修改 ```find_the_same.py``` 的路径参数，其中输入文件路径是在 “大模型 API” 流程中生成的数据集文件，然后运行这个 py 文件，切分子问题并筛选出两次答案相同的数据
```python
ifn = '/data/disk4/home/chenrui/ai-project/logical_reasoning/data/input/gpt/new_data_gpt4_25288_1.jsonl'  # 子问题拆分后的 jsonl 
ofn = '/data/disk4/home/chenrui/ai-project/logical_reasoning/data/input/gpt/gpt4_4590_806_wrong.jsonl'  # 输出文件路径
tfn = '/data/disk4/home/chenrui/ai-project/logical_reasoning/data/input/tmp/gpt4.jsonl'
```
2. 修改 ```jsonl_2_instruction.py``` 的路径参数，其中输入文件路径是上一步生成的筛选后的数据集文件，然后运行这个 py 文件，将 jsonl 格式转为大模型微调（指令监督微调）需要的指令格式
```python
input_file = '/data/disk4/home/chenrui/ai-project/logical_reasoning/data/input/gpt/gpt4_4590_806_split.jsonl'
output_file = '/data/disk4/home/chenrui/ai-project/logical_reasoning/data/input/gpt/gpt4_4590_806_inst.json'
tmp_file = '/data/disk4/home/chenrui/ai-project/logical_reasoning/data/input/tmp/tmp.jsonl'
tmp_dir = '/data/disk4/home/chenrui/ai-project/logical_reasoning/data/input/tmp'

need_split = False  # 是否需要切分子问题
task_type = 'raw'  # 任务类型：raw 代表使用原提示词模板
```

注：数据处理部分分成两步处理是因为切分子问题后的格式也是我们在测试中用到的重要数据格式，如需要一步到位，可以自行修改代码（代码逻辑非常简单），将处理逻辑合并到一个文件中。

### 微调
由于项目的硬件环境限制（磁盘 50GB，显存 32GB），我们考虑用 InternLM2.5-20B-Chat 量化后再进行 LoRA 微调。（显存占用在 25G 以内）

模型下载的方法：运行下面两行 python 代码（以 Qwen2-7B-Instruct 为例）
```python
from modelscope import snapshot_download
model_dir = snapshot_download('qwen/qwen2-7b-instruct')
```

#### InternLM2.5-20B-Chat QLoRA 微调（推荐）
这里我们用的是 QLoRA 微调（显存限制），因此需要对原来的预训练模型进行 bnb-8bit 量化，方法是修改路径参数后在终端中运行下面的代码。
```python
model_path = '/data/disk4/home/chenrui/.cache/modelscope/hub/qwen/qwen1___5-32b-chat'  # 预训练模型加载路径
save_path = '/data/disk4/home/chenrui/Qwen1.5-32B-Chat-bnb-4bit'  # 量化后的模型保存路径
```
```bash
CUDA_VISIBLE_DEVICES=0 python quantize.py
```

后面的流程只会用到量化后的模型（因此提交用于推理的代码包时也只需要包含量化后的模型）

输入：sft指令集、预训练模型（bnb-8bit 量化）和分词器
输出：模型检查点文件（包含模型权重、训练过程的指标等）

这里我们使用的是 llama-factory 训练框架，首先我们需要设置好训练的超参数，写在 bash 文件中，如下所示。
```bash
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train \
    --stage sft \
    --do_train True \
    --model_name_or_path /data/disk4/home/chenrui/Qwen1.5-32B-Chat-bnb-4bit \
    --preprocessing_num_workers 16 \
    --finetuning_type lora \
    --template qwen \
    --flash_attn auto \
    --dataset logical_problems_large \
    --cutoff_len 1024 \
    --learning_rate 1.0e-4 \
    --lr_scheduler_type cosine \
    --num_train_epochs 3.0 \
    --max_samples 10000 \
    --per_device_train_batch_size 8 \
    --val_size 0.1 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --max_grad_norm 1.0 \
    --evaluation_strategy steps \
    --logging_steps 50 \
    --eval_steps 100 \
    --save_steps 200 \
    --warmup_ratio 0.1 \
    --optim adamw_torch \
    --packing False \
    --report_to none \
    --output_dir saves/qwen1.5-32b-large/lora/sft \
    --fp16 True \
    --plot_loss True \
    --ddp_timeout 180000000 \
    --include_num_input_tokens_seen True \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0.2 \
    --lora_target all \
    --plot_loss True \
    --overwrite_cache True \
    --overwrite_output_dir True \
    # --deepspeed examples/deepspeed/ds_z3_config.json \
```
注：参数的含义可以在这里查看--[使用LlamaFactory进行模型微调：参数详解](https://blog.csdn.net/kjzd123/article/details/139794858)

另：若需要使用自定义数据集则需要把数据集放在 data/ 目录下，并在 ```dataset_info.json``` 按照格式加上该路径（具体操作见 data/ 目录下的 README_zh.md），然后在 bash 文件对应参数选择你的数据集名称。其他具体使用方法可以参考 llama-factory 项目中的 README_zh.md.

设置好参数后，直接在终端中用下面的命令运行上面的 bash 文件
```bash
bash train.sh
```
注意：启动项目时需要等待 1-2 分钟才会开始训练。

### 推理验证
微调结束后，我们获得了模型对应的 lora 权重，在这一步我们需要加载原模型和 lora 权重来对数据集进行验证。有两种验证的方式，一是修改路径后直接运行项目中的 ```validation.py``` 文件，二是使用 llama-factory 的 webchat 可视化推理功能。

#### 方法一：简单脚本推理验证
我们使用的验证数据集包含 5000 个样本（训练集不包含），可以根据需要随机选择一定数量的样本（我们的实验使用 1000 和 5000），然后修改 ```validation.py``` 中如下的路径参数，运行即可。
```python
# 模型和lora权重路径
model_path = '/data/disk4/home/chenrui/Qwen1.5-32B-Chat-bnb-4bit'
lora_path = '/data/disk4/home/chenrui/ai-project/logical_reasoning/LLaMA-Factory/saves/qwen1.5-32b-large-1/lora/sft/checkpoint-4000' 
# 验证数据集路径
input_file = '/data/disk4/home/chenrui/ai-project/logical_reasoning/data/input/validation_inst_5000.json'  
```
运行的方式是在文件的同一目录下终端中输入下面的命令（可以根据需要指定 GPU 设备）
```bash
python validation_rag.py
```
输入：原模型、训练得到的 lora 权重、数据集
输出：验证的正确率
  
#### 方法二：webchat 网页可视化推理验证
在终端中输入下面的命令（其中 GPU 设备可以根据需要设置）
```bash
CUDA_VISIBLE_DEVICES=1 llamafactory-cli webchat examples/inference/llama3_lora_sft.yaml
```

注意：这一步并非必须，只是为了严谨，若不需要可以跳过

### 生成测试集答案
这一步是用我们训练好的模型生成最终提交的测试集答案，只需要修改路径参数即可直接运行，运行的方式是在文件的同一目录下终端中输入下面的命令。
```bash
python submit.py
```
需要修改的路径参数如下：
```python
# 输入文件路径
input_file = '/data/disk4/home/chenrui/ai-project/logical_reasoning/data/input/raw/round1_test_data.jsonl'
# 用Qwen2-72b生成的基准答案文件路径
baseline_file_path = '/data/disk4/home/chenrui/ai-project/logical_reasoning/data/output/test/Qwen2-72B-test.jsonl'
# 输出文件路径
output_file = f'/data/disk4/home/chenrui/ai-project/logical_reasoning/data/output/submit/submit_{ID}.jsonl'
# 加载模型和LoRA权重
model_path = '/data/disk4/home/chenrui/Qwen1.5-32B-Chat-bnb-4bit'
lora_path = '/data/disk4/home/chenrui/LLaMA-Factory-main/saves/qwen1.5-32b-large/lora/sft/checkpoint-400'  # 这里改称你的 lora 输出对应 checkpoint 地址
```
运行结束后，除了保存用于提交的 jsonl 文件外，还会输出一个与基准答案的相似度，这个相似度可以用来评估生成答案的质量，具体的实验结果见下表。

## 实验结果
| 模型 | 是否量化 | 提示词 | 验证正确率 | 基答相似度 | 提交得分 | 硬件依赖 |
| -------------- | ---- | ----- | ----- | ----- | ------ | ------ | 
| Qwen2-72B(API) | - | raw | - | - | 0.8449* | - |
| Qwen2-72B-Instruct | bnb-int4 | raw | 0.8860 | 0.8682 | 0.8358* | 45G | 
| Qwen1.5-32B-Chat | bnb-int4 | raw | **0.8810** | 0.8125 | 0.7907 | 25G |
| Qwen2-7B-Instruct | - | raw | 0.8590 | 0.7959 | 0.7854 | 20G |
| Qwen2-Math-7B-Instruct | - | raw | 0.8000 | 0.7387 | - | 20G |
| InternLM2.5-20B-Chat | bnb-int8 | raw | 0.8780 | **0.8193** | 0.7982 | 25G |
| Qwen1.5-32B-Chat | bnb-int4 | 3-shot | 0.8060 | 0.7545 | - | 30G |
| InternLM2.5-20B-Chat | bnb-int8 | RAG(1-shot) | 0.8680 | 0.8193 | - | 25G |
| InternLM2.5-20B-Chat | bnb-int8 | RAG(3-shot) | 0.8890 | 0.8298 | 0.8042 | 25G |
| InternLM2.5-20B-Chat | bnb-int8 | **RAG(5-shot)** | 0.8710 | 0.8306 | **0.8117** | 25G |
| InternLM2.5-20B-Chat | bnb-int8 | RAG(7-shot) | - | 0.8215 | - | 25G |

注1：上述结果（除API外）都是基于 LoRA 微调后的模型
注2：* 表示不满足竞赛要求（32G 显存限制）
注3：验证集包含从验证数据集中随机筛选的 1000 条数据
注4：基准答案就是调用 Qwen2-72B 的 API 的结果（表中第一行）