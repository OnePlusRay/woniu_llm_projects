# 大模型应用：逻辑推理

用给定数据集（需要扩充）对大模型进行微调，使其在复杂逻辑推理问题中获得较好的表现。

## 整体流程
1. 调用 GPT4 的 API 接口扩充数据集（从 500 扩充到 10000 以上）
2. 调用其他大模型（如 Qwen2-72B、claude 3.5 等）的 API 检验生成的数据质量（对题目进行重做，若与原题目答案一致，则认为质量过关）
3. 数据清洗和格式转换（筛选上面一步中两次答案一致的数据，将生成的数据集转换为微调需要的指令格式）
4. 对 Qwen2-7B-Instruct 进行 LoRA 指令监督微调（或使用更大的模型量化），并对验证集进行正确率测试（训练集数据量为 5000 左右时训练用时约为 1 小时）
5. 用微调得到的 LoRA 权重与原模型合并后进行推理，生成最后提交的结果

## 项目难点
- 扩充数据集时，调用 GPT4 生成数据响应速度较慢，批量跑数据耗时比较严重（一条数据大约需要 1 分钟，10000 条则需要好几天，使用多线程可以有所改善），烧钱也比较厉害
- 微调时超参数的选定需要一些试错成本

## 文件结构
### 代码部分
- src/：调用大模型 API 的核心处理函数（这里面的函数都在 ```api_run.py```调用）
  - ```get_ans.py```：回答逻辑推理问题
    - ```produce```：调用大模型生成答案（训练数据）
    - ```produce_without_eval```：调用大模型生成答案（测试数据）
    - ```evaluate```：模型评估
    - ```get_ans```：主函数
  - ```get_ans_con.py```：多线程并发版本（速度快）
  - ```generate_problems.py```：调用 GPT4 生成数据
- data_processing/：数据处理相关函数
  - ```jsonl_2_instruction.py```：jsonl 格式转指令格式（可选是否需要拆分子问题）
  - ```split_jsonl.py```：对原数据集格式拆分子问题
  - ```find_the_same.py```：筛选出原问题答案和大模型生成答案一致的数据
  - ```inst_change_prompt.py```：更换指令集中的提示词
  - ```utils.py```：数据处理常用的辅助函数（生成提示词模板和提取答案）
- utils/：API 调用相关辅助函数
  - ```chain.py```：通过 LangChain 构建 prompt-llm-output 调用链
  - ```llm.py```：可选的大模型 API 函数
- baseline/：官方提供的基本代码和数据
- ```.env```：环境变量
- ```api_run.py```：调用大模型 API 运行脚本
- ```generate_problems```：调用大模型 API 生成数据，扩充数据集（后续要封装成函数，统一在 ```api_run``` 中运行）
- ```quantize.py```：模型量化
- ```test.py```：测试脚本
- ```train_lora.ipynb```：LoRA 微调训练 + 验证集测试（核心）
- ```inference_inst.py```：用微调的指令集来推理验证
- ```requirements.txt```：项目依赖

### 数据模型部分
- assets/：提示词模板
  - ```generate_problems.txt```：生成数据（生成新问题）
  - ```generate_questions.txt```：生成数据（扩充子问题）
  - ```logical_reasoning_prompt.txt```：大模型回答逻辑问题
- checkpoints/：模型检查点文件
- data/：输入和输出的数据（包括 jsonl 和 json 数据）
  - input/：输入数据
    - gpt/：用 GPT4 生成的新数据集
    - new/：人工检查和修改后的数据集
    - raw/：原数据集
    - tmp/：中间文件保存路径
  - output/：输出数据
    - submit/：用于提交的测试集结果
- logs/：TensorBoard 的日志文件
- output/：最终模型保存路径


## 项目运行
首先安装依赖
```bash
pip install -r requirements.txt
```
### 大模型 API 
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

### 数据准备
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

### LoRA 微调
由于项目的硬件环境限制（磁盘 50GB，显存 32GB），这里我们目前考虑以下两种方案：
- 用较小的模型（如 Qwen2-7B）直接进行 LoRA 微调
- 用稍大的模型（如 Qwen1.5-32B）量化后再进行 LoRA 微调

模型下载的方法：运行下面两行 python 代码（以 Qwen2-7B-Instruct 为例）
```python
from modelscope import snapshot_download
model_dir = snapshot_download('qwen/qwen2-7b-instruct')
```

#### Qwen2-7B-Instruct LoRA 微调
输入：sft指令集、预训练模型和分词器
输出：模型检查点文件（包含模型权重、训练过程的指标等）

设置好使用的 GPU 或 CPU 设备、文件输入输出路径后一键运行 ```train_lora.ipynb``` 即可。需要修改的路径参数如下：
```python
data_path = '/data/disk4/home/chenrui/ai-project/logical_reasoning/data/input/raw/round1_train_data_easy_inst.json'  # 数据集（指令集）路径
tmp_dir = '/data/disk4/home/chenrui/ai-project/logical_reasoning/data/input/tmp'  # 存放临时文件路径
model_path = '/data/disk4/home/chenrui/ai-project/Qwen2-7B-Instruct'  # 模型路径
```

注意下面这行代码会启动 tensorboard 并打断运行，若不需要可以注释掉，则可以实现一键微调+推理验证。
```python
!tensorboard --logdir=./logs --host=172.20.2.1
```
微调的基本方法参照 [Qwen2-7B-Instruct lora 微调](https://blog.csdn.net/xiaobing259/article/details/140594017)

主要超参数解释如下
```python
args = TrainingArguments(
    output_dir=f"./checkpoints/Qwen2_7B_instruct_lora",
    per_device_train_batch_size=4,  # 每个设备的训练数据批量
    per_device_eval_batch_size=4,  # 每个设备的验证数据批量
    num_train_epochs=2,  # 训练轮数
    logging_steps=10,  # 每训练多少步记录一次信息（可在 TensorBoard 查看）
    save_steps=1000,  # 每训练多少步保存一次模型检查点
    learning_rate=1e-5,  # 学习率
    save_on_each_node=True,  
    gradient_checkpointing=True,
    lr_scheduler_type="linear",  # 学习率调度器类型
    warmup_steps=500,
    gradient_accumulation_steps=2,  # 梯度累积步数，即每计算多少次梯度进行一次反向传播
    evaluation_strategy="steps",  # 设置评估策略
    eval_steps=500,  # 每训练多少步进行一次评估
    # fp16=True,  # 使用混合精度训练，防止梯度爆炸
    load_best_model_at_end=True,  # 在训练结束时加载最佳模型
    metric_for_best_model="eval_loss",  # 监控的指标
    greater_is_better=False,  # 因为我们希望损失最小化
    logging_dir='./logs',  # TensorBoard 日志目录
    report_to="tensorboard",  # 启用 TensorBoard
)
```
注：这个训练框架的效果一般，调整超参数后 loss（包括训练和验证） 最低在 0.23 左右，验证集的 acc 在 81% 左右，与直接调用 Qwen2-72B 的 api 的结果相似度为 76% 左右，用测试集生成答案提交的分数最高为 0.76 左右，不能满意。因此下面我们做以下两点调整，更换参数量更大的模型进行 4bit 量化，并用 llama-factory 框架来进行 qlora 微调。

#### Qwen1.5-32B-Chat-bnb-4bit QLoRA 微调（推荐）
这里我们用的是 QLoRA 微调（显存限制），因此需要对原来的预训练模型进行 bnb-4bit 量化，方法是在终端中运行下面的代码。
```bash
CUDA_VISIBLE_DEVICES=0 python quantize.py
```
需要修改的路径参数如下：
```python
# 预训练模型加载路径
model_path = '/data/disk4/home/chenrui/.cache/modelscope/hub/qwen/qwen1___5-32b-chat'
# 量化后的模型保存路径
save_path = '/data/disk4/home/chenrui/Qwen1.5-32B-Chat-bnb-4bit'
```
后面的流程只会用到量化后的模型（因此提交用于推理的代码包时也只需要包含量化后的模型）

输入：sft指令集、预训练模型（bnb-4bit 量化）和分词器
输出：模型检查点文件（包含模型权重、训练过程的指标等）

这里我们使用的是 llama-factory 训练框架，首先我们需要设置好训练的超参数，我用下面的 bash 文件的内容作为示例（yaml 文件中的参数设置基本同理，仅有格式上的差别）。
```bash
CUDA_VISIBLE_DEVICES=1 llamafactory-cli train \
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

设置完参数后，有三种运行项目的方法：
- 方法一：用下面的命令运行 yaml 文件（根据需要自行修改文件中的参数）
```bash
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/train_qlora/llama3_lora_sft_otfq.yaml  # 文件路径（具体参考 examples 文件夹中的 README）
```
- 方法二：直接在终端中用下面的命令运行上面的 bash 文件（推荐）
```bash
bash run_qlora_32b.sh
```
- 方法三：LLaMA Board 可视化微调（由 Gradio 驱动）
```bash
llamafactory-cli webui
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
python validation.py
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
运行结束后，除了保存用于提交的 jsonl 文件外，还会输出一个与基准答案的相似度，这个相似度可以用来评估生成答案的质量，根据我们的经验，0.86 的相似度可以获得 0.83 的分数，0.80 的相似度可以获得 0.79 的分数。

## 实验结果
| 模型 | 提示词 | 1000验证 | 5000验证 | submit相似度 | 提交得分 | 显存 |
| -------------- | ---- | ----- | ----- | ----- | ------ | ------ | 
| Qwen2-72B-Instruct-bnb-int4（qlora微调） | raw | - | - | 0.8682 | 0.8358* | 45G | 
| Qwen1.5-32B-Chat-bnb-int4（qlora微调） | raw  | 0.8610 | - | 0.8057 | 0.7899 | 25G |
| Qwen2-7B-Instruct（lora微调） | raw | 0.8590 | 0.8412 | 0.7959 |  0.7854 | 20G |
| Qwen2-Math-7B-Instruct（lora微调） | raw | 0.8000 | 0.7850 | 0.7387 | - | 20G |
| Qwen1.5-32B-Chat-bnb-int4（qlora微调） | 3-shot | 0.8060 | - | 0.7545 | - | 30G |

注：* 表示不满足竞赛要求（32G 显存限制）