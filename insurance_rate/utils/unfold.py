import pandas as pd
import re

patterns = [r'.*\d+~\d+(?=[/ ]|$).*', r'.*\d+-\d+(?=[/ ]|$).*', r'.*\d+-\d+岁(?=[/ ]|$).*', r'.*\d+-\d+周岁(?=[/ ]|$).*', r'.*\d+~\d+岁(?=[/ ]|$).*', r'.*\d+~\d+周岁(?=[/ ]|$).*', r'.*\d+岁-\d+岁(?=[/ ]|$).*', r'.*\d+周岁-\d+周岁(?=[/ ]|$).*', r'.*\d+岁~\d+岁(?=[/ ]|$).*', r'.*\d+周岁~\d+周岁(?=[/ ]|$).*', r'.*\d+至\d+(?=[/ ]|$).*', r'.*\d+至\d+(?=[/ ]|$).*', r'.*\d+岁至\d+岁(?=[/ ]|$).*', r'.*\d+周岁至\d+周岁(?=[/ ]|$).*', r'.*\d+至\d+岁(?=[/ ]|$).*', r'.*\d+至\d+周岁(?=[/ ]|$).*', r'.*\d+到\d+(?=[/ ]|$).*', r'.*\d+到\d+(?=[/ ]|$).*', r'.*\d+岁到\d+岁(?=[/ ]|$).*', r'.*\d+周岁到\d+周岁(?=[/ ]|$).*', r'.*\d+到\d+岁(?=[/ ]|$).*', r'.*\d+到\d+周岁(?=[/ ]|$).*']

#去除无关字符
def remove_specific_characters(input_string):
    # 定义正则表达式模式，匹配中文字符、英文字母和反斜杠
    pattern = r'[\u4e00-\u9fa5a-zA-Z/]'
    # 使用re.sub()函数替换匹配的字符为空字符串
    result = re.sub(pattern, '', input_string)
    return result

#将表格中的年龄区间展开为各年龄点。如：1-3岁，展开为1,2,3。共三行
def unfold_table(df):
    new_df = pd.DataFrame(columns = list(df.columns))
    print("DataFrame:")
    print(df)
    print("\n遍历每一行:")
    columns = df.columns
    for index, row in df.iterrows():
        row_content = row[columns[0]]
        contents = row_content.split('/')
        if all(all(not re.match(pattern, content) for pattern in patterns) for content in contents):
            new_df = new_df._append(row)
            continue
        for content in contents:
            #若满足正则表达式，逐行展开
            if any(re.match(pattern, content) for pattern in patterns):
                new_content = content.replace('至','-').replace('到','-').replace('~','-')
                num_range = remove_specific_characters(new_content)
                [num_1,num_2] = num_range.split('-')
                num_1 = int(num_1)
                num_2 = int(num_2)
                for num in range(num_1,num_2+1):
                    print(row_content,content)
                    column_1 = row_content.replace(content, str(num))
                    column_2 = row[columns[1]]
                    new_row = pd.Series({columns[0]: column_1, columns[1]: column_2})
                    new_df = new_df._append(new_row, ignore_index=True)
    return new_df

#将因子陈列中的年龄区间展开为年龄点
def unfold_enum_list(enum_list: list):
    new_enum_list = []
    if all(all(not re.match(pattern, enum) for pattern in patterns) for enum in enum_list):
        return enum_list
    for enum in enum_list:
        if any(re.match(pattern, enum) for pattern in patterns):
                new_content = enum.replace('至','-').replace('到','-').replace('~','-')
                num_range = remove_specific_characters(new_content)
                [num_1,num_2] = num_range.split('-')
                num_1 = int(num_1)
                num_2 = int(num_2)
                for num in range(num_1,num_2+1):
                    new_enum_list.append(str(num))
    return new_enum_list

if __name__ == "__main__":  
    data = {
        '保险方案/被保人年龄/被保人社保情况': ['标准版/0至4月/有社保','标准版/5~10/有社保'],
        'fee': [580, 233],
    }
    df = pd.DataFrame(data)
    new_df = unfold_table(df)
    print(list(new_df['fee']),'----------------')
    new_df['fee'] = [int(fee * 100) for fee in new_df['fee']]
    print(new_df)

