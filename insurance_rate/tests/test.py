import pdb


def post_process_row_title(table_structure:dict):
    '''
    大模型识别效果有可能不准，会出现row_title中间夹杂content的情况
    先把最后一个main_title到第一个row_title之间的都赋成row_title
    再把第一个row_title到最后一个row_title之间的都赋成row_title
    '''
    # return table_structure
    row_type = "row_title"
    type_list = [table_structure[index] for index in table_structure]  # 表格中每行结构的列表
    # pdb.set_trace()  # 这个断点展示了在后处理row_title之前第4列就已被识别为row_title
    if row_type not in type_list:  # 若没有row_title则无需处理
        return table_structure
    for index, label in table_structure.items():  
        if label == "row_title":  # 遇到row_title就退出循环
            break
        elif label == "main_title":  # 遇到main_title就跳过
            continue
        #print('*******************',index,label)
        table_structure[index] = "row_title"  # 若是content则将其结构类型改为row_title
    first_index = -1
    last_index = -1
    for i in range(len(type_list)):  # 找到第一个和最后一个row_title的位置
        if type_list[i] == row_type and i != len(type_list) - 1:  # 要求最后一个row_title的位置不是最后一行/列（否则大概率是空值）
            last_index = i
            if first_index < 0:
                first_index = i

    if type_list[len(type_list) - 1] == row_type:
        table_structure[str(len(type_list))] = 'additional_information'
    
    check = all(element == row_type for element in type_list[first_index:last_index + 1])  # 检查是否第一个和最后一个row_title之间全是row_title
    # pdb.set_trace()
    if check:
        return table_structure
    if first_index >= 0 and last_index >= 0:  # 若第一个和最后一个row_title之间不全是row_title，将中间的部分全部更新为row_title
        table_structure.update({f'{i}': "row_title" for i in range(first_index + 1, last_index + 2)})
    # pdb.set_trace()
    
    return table_structure

# table_structure = {'1': 'main_title', '2': 'row_title', '3': 'concent', '4': 'row_title', '5': 'content', '6': 'content', '7': 'content', '8': 'row_title'}
# print(table_structure)
# table_structure = post_process_row_title(table_structure)
# print(table_structure)

def parse_match_factor_res(match_factor_res):
    xml_res = parse_xml_output_without_thinking(match_factor_res, tags=["result"], first_item_only=True)
    match_factor = xml_res["result"]
    if match_factor is None:
        return
    return match_factor

import re

#去除thinking后解析
def parse_xml_output_without_thinking(output_str:str, tags:list, thinking_tag:str = 'thinking', first_item_only:bool = False):
    output_thinking=parse_xml_output(output_str, [thinking_tag], first_item_only)
    print('0000000',output_thinking,type(output_thinking))
    output_without_thinking = output_str
    thinking_contents = output_thinking.get(thinking_tag, "")
    if output_thinking.get(thinking_tag, ""):
        if isinstance(thinking_contents, list):
            for content in thinking_contents:
                output_without_thinking=output_str.replace(content,'').replace(f'<{thinking_tag}>','').replace(f'</{thinking_tag}>','').strip()
        elif isinstance(thinking_contents, str):
            output_without_thinking=output_str.replace(thinking_contents,'').replace(f'<{thinking_tag}>','').replace(f'</{thinking_tag}>','').strip()
    xml_dict = parse_xml_output(output_without_thinking, tags, first_item_only)
    return xml_dict

#解析xml结果
def parse_xml_output(output_str:str, tags:list, first_item_only=False):
    xml_dict = {}
    for tag in tags:
        texts = re.findall(rf'<{tag}>(.*?)</{tag}>', output_str, re.DOTALL)
        if texts:
            xml_dict[tag] = texts[0] if first_item_only else texts
    return xml_dict

x = parse_match_factor_res(match_factor_res="<thinking>根据给定的信息:- <factor>被保人性别</factor>- <factor_enum>男</factor_enum>- <factor_enum_mapping>中的选项为['男', '女']可以判断出<factor_enum>男</factor_enum>对应的分类是男。</thinking><result>男</result>")
print(x)

factor_enum_dict_list = [
	{'factor': '保险方案', 'enum': ['基础责任保费', '重疾住院津贴（可选）', '重大疾病保险金（可选）', '特定疾病医疗保险金52种轻症（可选）']}, 
	{'factor': '被保人年龄', 'enum': ['28天-4周岁', '5-10周岁', '11-13周岁', '14-15周岁', '16-20周岁', '21-25周岁', '26-30周岁', '31-35周岁', '36-40周岁', '41-45周岁', '46-50周岁', '51-55周岁', '56-60周岁', '61-65周岁', '66-70周岁', '71-75周岁', '76-80周岁', 	'81-85周岁', '86-90周岁', '91-95周岁', '96-105周岁']}, 
	{'factor': '被保人社保情况', 'enum': ['有社保', '无社保', '有/无社保']}
]
column = '保险方案'
matching_dicts = [d for d in factor_enum_dict_list if column in d.values()][0]['enum']

print(matching_dicts)

import re

def chinese_to_arabic(chinese_str):
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

# 示例字符串
text = '5百'

# 提取并转换中文数字
arabic_number = extract_and_convert_chinese_number(text)
print(arabic_number)  # 输出: 14

def convert_to_number(chinese_string:str):
    '''保险保额的金额单位匹配'''
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
    
print(convert_to_number(text))
pattern = re.compile(r'\d+')
is_num = re.search(pattern, text)
print(bool(is_num))

# 精度问题
# a = '155.7'
# print(float(a) * 100)
# print(int(round(a * 100)))

import pandas as pd
import openpyxl

# # 创建一个示例 DataFrame
# data = {
#     'A': [1, 2, 3],
#     'B': [4, 5, 6]
# }
# df = pd.DataFrame(data)

# # 创建一个 ExcelWriter 对象
# file_path = "/sdata/chenrui/ai-project/insurance_rate/data/output/example.xlsx"
# with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
#     # 写入第一个工作表
#     df.to_excel(writer, sheet_name='Sheet1')
#     writer.book.save(file_path)
#     print("Sheet 'Sheet1' 已保存")

#     # 修改 DataFrame 并写入第二个工作表
#     df['A'] = df['A'] * 2
#     df.to_excel(writer, sheet_name='Sheet2')
#     writer.book.save(file_path)
#     print("Sheet 'Sheet2' 已保存")

#     # 修改 DataFrame 并写入第三个工作表
#     df['B'] = df['B'] * 3
#     df.to_excel(writer, sheet_name='Sheet3')
#     writer.book.save(file_path)
#     print("Sheet 'Sheet3' 已保存")

# print("所有工作表处理完毕")

# base_table = pd.read_excel("/sdata/chenrui/ai-project/insurance_rate/data/output/达尔文9号重疾险费率表1_flatten_1718359596.xlsx")
# factor_column = base_table['5-PLAN/5-AMOUNT/5-PAYMENTNUM/5-INSTERMNUM/2-SEX/2-AGE']
# factor_list = '5-PLAN/5-AMOUNT/5-PAYMENTNUM/5-INSTERMNUM/2-SEX/2-AGE'.split('/')
# factor_column_split = factor_column.str.split('/', expand=True)  # 使用str.split方法拆分每个元素，并将结果转换为DataFrame
# factor_column_split.columns = factor_list  # 为DataFrame的列命名
# factor_column_split['5-AMOUNT'] = '1000'

# # 保存修改后的 DataFrame 到新的 Excel 文件

# base_table['5-PLAN/5-AMOUNT/5-PAYMENTNUM/5-INSTERMNUM/2-SEX/2-AGE'] = factor_column_split.apply(lambda row: '/'.join(row.astype(str)), axis=1)
# base_table.to_excel('/sdata/chenrui/ai-project/insurance_rate/data/output/达尔文9号.xlsx', index=False)  # 请将 'output.xlsx' 替换为你想要保存的文件名

# table = pd.read_excel("/sdata/chenrui/ai-project/insurance_rate/data/real_in/达尔文9号样本.xlsx")
# print(table)
# element_types = table.applymap(type)
# print(element_types)

# try:
#     # 尝试将一个非数字字符串转换为整数
#     num = int("not_a_number")
# except ValueError as e:
#     # 捕获 ValueError 异常并获取异常信息
#     print(f"转换失败：{e}")

# import json

# # 示例列表
# data = [
# 	{'factor': '保险方案', 'enum': ['基础责任保费', '重疾住院津贴（可选）', '重大疾病保险金（可选）', '特定疾病医疗保险金52种轻症（可选）']}, 
# 	{'factor': '被保人年龄', 'enum': ['28天-4周岁', '5-10周岁', '11-13周岁', '14-15周岁', '16-20周岁', '21-25周岁', '26-30周岁', '31-35周岁', '36-40周岁', '41-45周岁', '46-50周岁', '51-55周岁', '56-60周岁', '61-65周岁', '66-70周岁', '71-75周岁', '76-80周岁', 	'81-85周岁', '86-90周岁', '91-95周岁', '96-105周岁']}, 
# 	{'factor': '被保人社保情况', 'enum': ['有社保', '无社保', '有/无社保']}
# ]

# # 保存列表到文件
# with open('data.json', 'w') as file:
#     json.dump(data, file)

# with open('data.json', 'r') as file:
#     loaded_data = json.load(file)

# print(loaded_data)
# import pandas as pd
# import pickle

# # 示例 DataFrame 列表
# df1 = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
# df2 = pd.DataFrame({'A': [7, 8, 9], 'B': [10, 11, 12]})
# dataframes = [df1, df2]

# # 保存 DataFrame 列表到文件



# import os
# import json
# def save_base_table_dict(base_table_dict: dict, file_name: str):
#     save_path = '/sdata/chenrui/ai-project/insurance_rate/data/save_files'
#     full_path = os.path.join(save_path, file_name)
#     with open(full_path, 'wb') as file:
#         pickle.dump(base_table_dict, file)
#     print(f"base_table_dict 已保存为 '{full_path}'。")

# def load_base_table_dict(file_path: str):
#     with open(file_path, 'rb') as file:
#         return pickle.load(file)


# import pandas as pd

# # 创建示例 DataFrame
# data = {
#     '因子组合': ['组合1', '组合2', '组合3'],
#     '费率': [0.1, 0.2, 0.3]
# }
# df = pd.DataFrame(data)

# # 创建示例因子可选项字典列表
# factor_enum_dict_list = [
#     {'factor': '因子1', 'enum': ['选项1', '选项2', '选项3']},
#     {'factor': '因子2', 'enum': ['选项A', '选项B', '选项C']}
# ]

# # 创建最终的数据结构
# example_data = {
#     'Sheet1': {
#         'base_table': df,
#         'factor_enum_dict_list': factor_enum_dict_list
#     }, 
#     'Sheet2': {
#         'base_table': df,
#         'factor_enum_dict_list': factor_enum_dict_list
#     }
# }

# # 打印示例数据结构
# # print(example_data)
# # save_base_table_dict(example_data, '计划_1')


# import pickle
# import openpyxl
# import pandas as pd

# def load_base_table_dict(file_path: str):
#     with open(file_path, 'rb') as file:
#         return pickle.load(file)

# base_table_dict = load_base_table_dict('/sdata/chenrui/ai-project/insurance_rate/data/save_files/保险保障责任')['保险保障责任']

# base_table = base_table_dict['base_table']
# factor_enum_dict_list = base_table_dict["factor_enum_dict_list"]
# print(factor_enum_dict_list)

import pandas as pd
df = pd.read_excel('/sdata/chenrui/ai-project/insurance_rate/data/output/君龙人寿样本_flatten_1718677139.xlsx')
print(df.iloc[89:110].applymap(type))

