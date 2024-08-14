# 导入所需的库
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import streamlit as st
from peft import PeftModel
import json
import pandas as pd

# 创建一个标题和一个副标题
st.title("💬 Yuan2.0 AI简历助手")

# 源大模型下载
from modelscope import snapshot_download
model_dir = snapshot_download('IEITYuan/Yuan2-2B-Mars-hf', cache_dir='./')

# 定义模型路径
path = './IEITYuan/Yuan2-2B-Mars-hf'
lora_path = './output/Yuan2.0-2B_lora_bf16/checkpoint-51'

# 定义模型数据类型
torch_dtype = torch.bfloat16 # A10
# torch_dtype = torch.float16 # P100

# 定义一个函数，用于获取模型和tokenizer
@st.cache_resource
def get_model():
    print("Creat tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(path, add_eos_token=False, add_bos_token=False, eos_token='<eod>')
    tokenizer.add_tokens(['<sep>', '<pad>', '<mask>', '<predict>', '<FIM_SUFFIX>', '<FIM_PREFIX>', '<FIM_MIDDLE>','<commit_before>','<commit_msg>','<commit_after>','<jupyter_start>','<jupyter_text>','<jupyter_code>','<jupyter_output>','<empty_output>'], special_tokens=True)

    print("Creat model...")
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch_dtype, trust_remote_code=True).cuda()
    model = PeftModel.from_pretrained(model, model_id=lora_path)

    return tokenizer, model

# 加载model和tokenizer
tokenizer, model = get_model()


template = '''
# 任务描述
假设你是一个AI简历助手，能从简历中识别出所有的命名实体，并以json格式返回结果。

# 任务要求
实体的类别包括：姓名、国籍、种族、职位、教育背景、专业、组织名、地名。
返回的json格式是一个字典，其中每个键是实体的类别，值是一个列表，包含实体的文本。

# 样例
输入：
张三，男，中国籍，工程师
输出：
{"姓名": ["张三"], "国籍": ["中国"], "职位": ["工程师"]}

# 当前简历
query

# 任务重述
请参考样例，按照任务要求，识别出当前简历中所有的命名实体，并以json格式返回结果。
'''


# 在聊天界面上显示模型的输出
st.chat_message("assistant").write(f"请输入简历文本：")


# 如果用户在聊天输入框中输入了内容，则执行以下操作
if query := st.chat_input():

    # 在聊天界面上显示用户的输入
    st.chat_message("user").write(query)

    # 调用模型
    prompt = template.replace('query', query).strip()
    prompt += "<sep>"
    inputs = tokenizer(prompt, return_tensors="pt")["input_ids"].cuda()
    outputs = model.generate(inputs, do_sample=False, max_length=1024) # 设置解码方式和最大生成长度
    output = tokenizer.decode(outputs[0])
    response = output.split("<sep>")[-1].replace("<eod>", '').strip()

    # 在聊天界面上显示模型的输出
    st.chat_message("assistant").write(f"正在提取简历信息，请稍候...")

    st.chat_message("assistant").table(pd.DataFrame(json.loads(response)))
