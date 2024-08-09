import pickle
import openpyxl
import pandas as pd

def load_base_table_dict(file_path: str):
    with open(file_path, 'rb') as file:
        return pickle.load(file)

base_table_dict = load_base_table_dict('/sdata/chenrui/ai-project/insurance_rate/data/save_files/保障责任')

print(base_table_dict)

# base_table = base_table_dict['base_table']
# factor_enum_dict_list = base_table_dict["factor_enum_dict_list"]

# # base_table['费率'] = base_table['费率'].astype(float)
# # base_table['费率'] = round(base_table['费率'] * 100)
# # base_table['费率'] = base_table['费率'].astype(int)


# print(base_table)
# print(factor_enum_dict_list)

# 将DataFrame保存为Excel文件
# base_table.to_excel('/sdata/chenrui/ai-project/insurance_rate/data/output/超级玛丽11号.xlsx', index=False, engine='openpyxl')
# print('已成功保存！')
