import re
import pdb
import pandas as pd
from utils.chain import get_chain
from utils.parse_xml import parse_xml_output, parse_xml_output_without_thinking
from utils.custom_chat_openai import input_data
import copy

# 需要用到的大模型调用链
factor_mapping_chain = get_chain("match_factor_enum")

# 辅助变量
factor_mapping = {
    "投保人年龄": "1-AGE",
    "被保人年龄": "2-AGE",
    "投保人性别": "1-SEX",
    "被保人性别": "2-SEX",
    "被保人社保情况": "2-MEDICALINSURANCE",
    "保险缴费期间": "5-PAYMENTNUM",
    "保险保障期间": "5-INSTERMNUM",
    "保险保额": "5-AMOUNT",
    "保险方案": "5-PLAN",
    "保险免赔额": "5-DUTY"
}


factor_enum_mapping = {
    "投保人性别": ["男", "女"],
    "被保人性别": ["男", "女"],
    "被保人社保情况": ["有", "无"],  # "有/无" 怎么处理
    "保险缴费期间": ["一次交清", "交n年", "交至n岁"],
    "保险保障期间": ["保终身", "保n年", "保至n岁"]
}

# 辅助函数
def chinese_to_arabic(chinese_str):
    '''中文数字转化为阿拉伯数字'''
    chinese_num_map = {
        '零': 0, '一': 1, '二': 2, '两': 2, '三': 3, '四': 4, '五': 5, '六': 6, '七': 7, '八': 8, '九': 9,
        '十': 10, '百': 100, '千': 1000, '万': 10000, '亿': 100000000
    }
    
    def convert_section(section):
        result = 0
        unit = 1
        num = 0
        for char in reversed(section):
            if char in chinese_num_map:
                val = chinese_num_map[char]
                if val >= 10:
                    if val > unit:
                        unit = val
                    else:
                        unit *= val
                else:
                    num = val * unit
                    result += num
            else:
                raise ValueError(f"Invalid character '{char}' in input string.")
        
        # 处理"十"的特殊情况，例如"十四"应该是14而不是4
        if '十' in section and section.index('十') == 0:
            result += 10
        return result
    
    result = 0
    sections = chinese_str.split('亿')
    for i, section in enumerate(sections):
        if section:
            result += convert_section(section) * (100000000 ** (len(sections) - i - 1))
    return result

def extract_and_convert_chinese_number(text):
    # 定义正则表达式模式，匹配中文数字
    pattern = r'[零一二两三四五六七八九十百千万亿]+'
    
    # 使用re.search()函数查找匹配的中文数字部分
    match = re.search(pattern, text)
    
    # 如果找到匹配的中文数字部分，进行转换
    if match:
        chinese_number = match.group(0)
        arabic_number = chinese_to_arabic(chinese_number)
        return arabic_number
    else:
        return None

def convert_to_number(chinese_string:str):
    '''保险保额的金额匹配'''
    chinese_string = chinese_string.replace(",", "")  # 匹配前先清除多余的符号，否则会匹配错误（如：1,000元要先转化为1000元） 
    pattern = re.compile(r'\d+')
    is_num = re.search(pattern, chinese_string)
    if is_num:  # 如果原数据包含数字，则只需匹配单位
        unit_to_value = {'十': 10, '百': 100, '千': 1000, '万': 10000, '亿': 100000000}
        pattern = re.compile(r'(\d+)([万亿百十]?)')
        match = pattern.search(chinese_string)
        if match:
            number, unit = match.groups()
            if unit:
                return str(int(number) * unit_to_value[unit])
            else:
                return number
        else:
            return chinese_string
    else:  # 若不包含数字，直接用中文数字转阿拉伯数字函数
        return str(extract_and_convert_chinese_number(chinese_string))
    
def payment_num(chinese_string:str, classification:str):
    '''保险缴费期间匹配'''
    pattern = re.compile(r'\d+')
    is_num = re.search(pattern, chinese_string)
    x = -1  # 默认 一次交清
    y = -1
    # pdb.set_trace()
    if is_num:  # 先考虑数字匹配
        if classification == '交n年':  # 交n年
            x = 1
            y = str(pattern.search(chinese_string).group())
        elif classification == '交至n岁':  # 交至n岁
            x = 2
            y = str(pattern.search(chinese_string).group())
    else:  # 再考虑形如“十五年交”的中文数字匹配
        if classification == '交n年':  # 交n年
            x = 1
            y = str(extract_and_convert_chinese_number(chinese_string))
        elif classification == '交至n岁':  # 交至n岁
            x = 2
            y = str(extract_and_convert_chinese_number(chinese_string))
    return f"{x}_{y}"

def instrem_num(chinese_string:str, classification:str):
    '''保险保障期间匹配'''
    pattern = re.compile(r'\d+')
    is_num = re.search(pattern, chinese_string)
    x = 3  # 默认 保终身
    y = 0
    if is_num:
        if classification == '保n年':  # 保n年
            x = 1
            y = str(pattern.search(chinese_string).group())
        elif classification == '保至n岁':  # 保至n岁
            x = 2
            y = str(pattern.search(chinese_string).group())
    else:
        if classification == '保n年':  # 保n年
            x = 1
            y = str(extract_and_convert_chinese_number(chinese_string))
        elif classification == '保至n岁':  # 保至n岁
            x = 2
            y = str(extract_and_convert_chinese_number(chinese_string))
    return f"{x}_{y}"

def match_factor_enum(factor_enum, factor_enum_mapping):
    '''用大模型判断各因子选项所属分类'''
    chain = factor_mapping_chain
    input_data.info = factor_enum
    match_res = chain.invoke({
        "factor_enum": factor_enum,
        "factor_enum_mapping": factor_enum_mapping
    })
    print(f"============>factor_enum: {factor_enum}")
    print(f"============>match_res: {match_res}")
    return match_res

def parse_match_factor_res(match_factor_res):
    '''解析大模型输出结果，只输出{result}'''
    xml_res = parse_xml_output_without_thinking(match_factor_res, tags=["result"], first_item_only=True)
    match_factor = re.sub(r'\s+', '', xml_res["result"])
    if match_factor is None:
        return
    return match_factor

def factor_name_classification_llm(sheet_info:dict):
    '''用大模型对需要分类匹配的因子选项进行分类'''
    base_table = sheet_info["base_table"]
    factor_enum_dict_list = sheet_info["factor_enum_dict_list"]

    # 先处理factor_enum_dict_list
    factor_enum_match_list = []  # 用于储存大模型对因子选项的分类结果
    factor_enum_match_list = copy.deepcopy(factor_enum_dict_list)  # 这里要用深拷贝，否则原列表也会被修改
    for i in range(len(factor_enum_match_list)):
        factor = factor_enum_dict_list[i]["factor"]
        if factor in factor_mapping:  
            factor_enum_match_list[i]['factor'] = factor_mapping[factor]

    # pdb.set_trace()
    for i in range(len(factor_enum_dict_list)):
        factor = factor_enum_dict_list[i]["factor"]
        factor_enum = factor_enum_dict_list[i]["enum"]

        # 因子名称--简单映射
        if factor in factor_mapping:  
            factor_enum_dict_list[i]['factor'] = factor_mapping[factor]

        # 因子选项
        if factor in factor_enum_mapping: 
            # 用大模型对因子选项进行分类
            for j in range(len(factor_enum)):
                match_factor_res = match_factor_enum(factor_enum[j], factor_enum_mapping[factor])
                # pdb.set_trace()
                factor_enum_match_list[i]['enum'][j] = parse_match_factor_res(match_factor_res)
                # pdb.set_trace()
                # 目前问题：factor_enum_dict_list通过大模型时也会被更新

    factor_list = []  # 包含的因子列表
    for factor_enum_dict in factor_enum_dict_list:
        factor_list.append(factor_enum_dict['factor'])

    # 再处理base_table（根据前面的factor_enum_dict_list和factor_enum_match_list）
    factor_column = base_table['因子组合']
    # pdb.set_trace()
    factor_column_split = factor_column.str.split('/', expand=True)  # 使用str.split方法拆分每个元素，并将结果转换为DataFrame
    # pdb.set_trace()
    factor_column_split.columns = factor_list  # 为DataFrame的列命名
    factor_column_split_match = copy.deepcopy(factor_column_split)  # 用于储存大模型对因子选项的分类结果

    # 把factor_enum_mapping的key更新成英文
    updated_factor_enum_mapping = {}
    for key, value in factor_enum_mapping.items():
        new_key = factor_mapping[key]
        updated_factor_enum_mapping[new_key] = value

    # 重新取一下factor_enum_dict_list
    # factor_enum_dict_list = sheet_info["factor_enum_dict_list"]

    for column in factor_column_split:
        # pdb.set_trace()
        if column in updated_factor_enum_mapping:
            # pdb.set_trace()
            dic = [d for d in factor_enum_dict_list if column in d.values()][0]['enum'] 
            dic_match = [d for d in factor_enum_match_list if column in d.values()][0]['enum']
            # pdb.set_trace()
            for index, value in factor_column_split[column].items():
                pos = dic.index(factor_column_split.loc[index, column])
                factor_column_split_match.loc[index, column] = dic_match[pos]
                
                
    # base_table['因子组合'] = factor_column_split.apply(lambda row: '/'.join(row.astype(str)), axis=1)
    # sheet_info["base_table"] = base_table
    # pdb.set_trace()
    return base_table, factor_column_split, factor_column_split_match, factor_enum_dict_list, factor_enum_match_list

def map_factor_name(sheet_info:dict):
    # 调用
    base_table, factor_column_split, factor_column_split_match, factor_enum_dict_list, factor_enum_match_list = factor_name_classification_llm(sheet_info)
    # 根据factor_column_split_match的内容更新factor_column_split
    for column in factor_column_split:
        for index, value in factor_column_split[column].items():
            if column == '1-SEX' or column == '2-SEX':
                factor_column_split.loc[index, column] = 'M' if factor_column_split_match.loc[index, column] == '男' else 'F'
            elif column == "2-MEDICALINSURANCE":  # 被保人社保情况
                factor_column_split.loc[index, column] = '1' if factor_column_split_match.loc[index, column] == '有' else '2'
            elif column == '5-PAYMENTNUM':  # 保险缴费期间
                factor_column_split.loc[index, column] = payment_num(factor_column_split.loc[index, column], factor_column_split_match.loc[index, column])
                # pdb.set_trace()
            elif column == '5-INSTERMNUM':  # 保险保障期间
                factor_column_split.loc[index, column] = instrem_num(factor_column_split.loc[index, column], factor_column_split_match.loc[index, column])
            elif column == '5-AMOUNT':  # 保险保额
                factor_column_split.loc[index, column] = convert_to_number(factor_column_split.loc[index, column])
            # pdb.set_trace()
    base_table['因子组合'] = factor_column_split.apply(lambda row: '/'.join(row.astype(str)), axis=1)
    sheet_info["base_table"] = base_table

    # 根据factor_enum_match_list的内容更新factor_enum_dict_list
    for i in range(len(factor_enum_dict_list)):
        factor = factor_enum_dict_list[i]["factor"]
        factor_enum = factor_enum_dict_list[i]["enum"]
        for j in range(len(factor_enum_dict_list[i]['enum'])):
            # 因子选项分类--只需简单分类
            if factor == '1-SEX' or factor == '2-SEX':
                factor_enum_dict_list[i]['enum'][j] = 'M' if factor_enum_match_list[i]['enum'][j] == '男' else 'F'
            elif factor == "2-MEDICALINSURANCE":
                factor_enum_dict_list[i]['enum'][j] = '1' if factor_enum_match_list[i]['enum'][j] == '有' else '2'

            # 因子选项分类--需要正则匹配
            elif factor == '5-PAYMENTNUM':  # 保险缴费期间
                factor_enum_dict_list[i]['enum'][j] = payment_num(factor_enum_dict_list[i]['enum'][j], factor_enum_match_list[i]['enum'][j])
                # pdb.set_trace()
            elif factor == '5-INSTERMNUM':  # 保险保障期间
                factor_enum_dict_list[i]['enum'][j] = instrem_num(factor_enum_dict_list[i]['enum'][j], factor_enum_match_list[i]['enum'][j])
            elif factor == '5-AMOUNT':  # 保险保额
                factor_enum_dict_list[i]['enum'][j] = convert_to_number(factor_enum_dict_list[i]['enum'][j])
    sheet_info["factor_enum_dict_list"] = factor_enum_dict_list
    # pdb.set_trace()
    return 


def batch_map_factor_name(base_table_dict: dict):
    '''将被调用的主函数：批量因子名称映射'''
    for sheet_name, sheet_info in base_table_dict.items():
        map_factor_name(sheet_info)
        # base_table = sheet_info["base_table"]  # 后处理，清除fee为空的行
        # base_table = base_table[base_table['费率'] != '']
