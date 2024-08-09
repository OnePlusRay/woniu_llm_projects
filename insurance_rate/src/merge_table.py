import pandas as pd
from src.table import Table
from utils.unfold import unfold_table, unfold_enum_list

#合并base_table_dict
def merge_base_table_dict(base_table_dict: dict, base_name: str, table: Table):
    '''
    base_table_dict = {'base_name':{'base_table':base_table, 'factor_enum_dict_list': factor_enum_dict_list}}
    base_table: 平铺后的费率表的基本表，只有两列：因子组合、fee
    factor_enum_dict_list：base_table中出现的factor_enum_dict列表
    factor_enum_dict = {'factor_1': enum_1,'factor_2': enum_2}
    factor_n：第n个因子名，如：被保人性别
    enum_n：第n个因子对应的遍历内容列表，如：['男','女']
    '''
    if base_name not in base_table_dict:
            base_table_dict[base_name]= {'base_table':table.base_table, 'factor_enum_dict_list': table.factor_enum_dict_list}
            print('-------------',table.base_table)
            print('-------------',table.factor_enum_dict_list)
            #pdb.set_trace()
    else:
        #更新factor_enum_dict_list
        old_factor_enum_dict_list = base_table_dict[base_name]['factor_enum_dict_list']
        new_factor_enum_dict_list = table.factor_enum_dict_list
        old_factor_list = [old_factor_enum['factor'] for old_factor_enum in old_factor_enum_dict_list]
        for new_factor_enum_dict in new_factor_enum_dict_list:
            new_factor = new_factor_enum_dict['factor']
            if new_factor not in old_factor_list:
                old_factor_enum_dict_list.append({'factor': new_factor, 'enum':new_factor_enum_dict['enum']})
                continue
            for old_factor_enum_dict in old_factor_enum_dict_list:
                if new_factor != old_factor_enum_dict['factor']:
                    continue
                for new_enum in new_factor_enum_dict['enum']:
                    if new_enum not in old_factor_enum_dict['enum']:
                        old_factor_enum_dict['enum'].append(new_enum)
        #将同样base_name的base_table合并
        base_table_dict[base_name]= {'base_table':pd.concat([base_table_dict[base_name]['base_table'],table.base_table],axis=0, ignore_index=True), 'factor_enum_dict_list': old_factor_enum_dict_list}
    return base_table_dict

#将原费率表中费率*100转换单位为分
def convert_and_multiply(lst):
    def try_convert_and_multiply(element):
        try:
            return int(round(float(element) * 100))  # 解决155.7 * 100 = 15569的问题：加round函数
        except ValueError:
            return element
    return [try_convert_and_multiply(item) for item in lst]

#合并各sheet
def merge_table_n_write(base_table_dict: dict, writer):
    for base_name in base_table_dict:
        try: 
            # print('^^^^^^^^^^^^^^^^^^^^^^^^',base_name)

            # 构建enum_df :展示各因子及其对应全部可选项
            table_enums_dict = base_table_dict[base_name]
            base_table = table_enums_dict['base_table']
            factor_enum_dict_list = table_enums_dict['factor_enum_dict_list']

            print('-------------',base_table)
            print('-------------',factor_enum_dict_list)

            factor_enum_infos = []
            factor_list = []

            #合并因子可选项 
            for factor_enum_dict in factor_enum_dict_list:
                factor_list.append(factor_enum_dict['factor'])
                enum_list = factor_enum_dict['enum']
                enum_list = unfold_enum_list(enum_list)
                concat_enum = ",".join(enum_list)
                factor_enum_info = f"{factor_enum_dict['factor']}:{concat_enum}"
                factor_enum_infos.append(factor_enum_info)
            # print(factor_enum_infos)
            combination = "/".join(factor_list)
            base_table.columns = [combination, 'fee']
            base_table = unfold_table(base_table)
            base_table['fee'] = convert_and_multiply(list(base_table['fee']))
            enum_df = pd.DataFrame(factor_enum_infos, columns=["因子可选项"])
            # print(enum_df)

            # 合并base_table与enum_df
            final_df = pd.concat([base_table, enum_df], axis=1)
            final_df.to_excel(writer, base_name, index=False)
            # print(final_df)
        except Exception as e:
            print(e)
            continue
    return writer
