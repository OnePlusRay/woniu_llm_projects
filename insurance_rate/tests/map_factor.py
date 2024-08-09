import re
import pdb


# 辅助变量
factor_mapping = {
    "投保人年龄": "AGE",
    "被保人年龄": "AGE",
    "投保人性别": "SEX",
    "被保人性别": "SEX",
    "被保人社保情况": "MEDICALINSURANCE",
    "保险缴费期间": "PAYMENTNUM",
    "保险保障期间": "INSTERMNUM",
    "保险保额": "AMOUNT",
    "保险方案": "PLAN",
    "保险免赔额": "DUTY"
}


factor_enum_mapping = {
    "投保人性别": ["男", "女"],
    "被保人性别": ["男", "女"],
    "被保人社保情况": ["有", "无"],  # 有/无的情况怎么处理（目前按"有"处理）
    "保险缴费期间": ["一次交清", "交n年", "交至n岁"],
    "保险保障期间": ["保终身", "保n年", "保至n岁"]
}

# 辅助函数
def convert_to_number(chinese_string):
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

def map_factor_name(sheet_info:dict):
    '''单个sheet的因子名称映射'''
    base_table = sheet_info["base_table"]
    factor_enum_dict_list = sheet_info["factor_enum_dict_list"]
    # factor_list = []
    # factor_enum_list = []
    for factor_enum_dict in factor_enum_dict_list:
        factor = factor_enum_dict["factor"]
        factor_enum = factor_enum_dict["enum"]
        # print(factor, factor_enum)

        # 因子名称直接映射
        factor_enum_dict['factor'] = factor_mapping[factor]
        # factor_list.append(factor)
        # pdb.set_trace()
        # print(factor, factor_enum)
        
        # 因子选项分类--简单映射
        if factor == '投保人性别' or factor == '被保人性别':
            for i in range(len(factor_enum_dict['enum'])):
                factor_enum_dict['enum'][i] = 'M' if '男' in factor_enum_dict['enum'][i] else 'F'

        elif factor == "被保人社保情况":
            for i in range(len(factor_enum_dict['enum'])):
                factor_enum_dict['enum'][i] = '1' if '有' in factor_enum_dict['enum'][i] else '2'
        
        # 因子选项分类--需要正则匹配
        elif factor == '保险缴费期间':
            pattern = re.compile(r'\d+')
            for i in range(len(factor_enum_dict['enum'])):
                x = -1  # 默认 一次交清
                y = -1
                if '年' in factor_enum_dict['enum'][i]:  # 交n年
                    x = 1
                    y = str(pattern.search(factor_enum_dict['enum'][i]).group())
                elif '岁' in factor_enum_dict['enum'][i]:  # 交至n岁
                    x = 2
                    y = str(pattern.search(factor_enum_dict['enum'][i]).group())
                factor_enum_dict['enum'][i] = f"{x}_{y}"

        elif factor == '保险保障期间':
            pattern = re.compile(r'\d+')
            for i in range(len(factor_enum_dict['enum'])):
                x = 3  # 默认 保终身
                y = 0
                if '年' in factor_enum_dict['enum'][i]:  # 保n年
                    x = 1
                    y = str(pattern.search(factor_enum_dict['enum'][i]).group())
                elif '岁' in factor_enum_dict['enum'][i]:  # 保至n岁
                    x = 2
                    y = str(pattern.search(factor_enum_dict['enum'][i]).group())
                factor_enum_dict['enum'][i] = f"{x}_{y}"

        elif factor == '保险保额':
            for i in range(len(factor_enum_dict['enum'])):
                factor_enum_dict['enum'][i] = convert_to_number(factor_enum_dict['enum'][i])  

    return


def batch_map_factor_name(base_table_dict:dict):
    '''将被调用的主函数：批量因子名称映射'''
    for sheet_name, sheet_info in base_table_dict.items():
        map_factor_name(sheet_info)

if __name__ == "__main__":
    base_table_dict = {
        '费率1': {
            'base_table': '表1',                                 
            'factor_enum_dict_list': [
                {'factor': '保险方案', 'enum': ['基础责任保费', '重疾住院津贴（可选）', '重大疾病保险金（可选）', '特定疾病医疗保险金52种轻症（可选）']}, 
                {'factor': '被保人年龄', 'enum': ['28天-4周岁', '5-10周岁', '11-13周岁', '14-15周岁', '16-20周岁', '21-25周岁', '26-30周岁', '31-35周岁', '36-40周岁', '41-45周岁', '46-50周岁', '51-55周岁', '56-60周岁', '61-65周岁', '66-70周岁', '71-75周岁', '76-80周岁', 	'81-85周岁', '86-90周岁', '91-95周岁', '96-105周岁']}, 
                {'factor': '被保人社保情况', 'enum': ['有社保', '无社保', '有/无社保']}
            ]
        },
        '费率2': {
            'base_table': '表2',                                 
            'factor_enum_dict_list': [
                {'factor': '保险缴费期间', 'enum': ['一次交清', '20年交', '交至70岁']}, 
                {'factor': '保险保障期间', 'enum': ['保至70岁', '保终身']},
                {'factor': '保险保额', 'enum': ['1万元', '20000元']}
            ]
        }
    }
    batch_map_factor_name(base_table_dict)
    # pdb.set_trace()
    print(base_table_dict)