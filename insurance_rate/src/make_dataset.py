import pymysql
import os
import re
from utils.parse_xml import parse_xml_output_without_thinking
import xml.etree.ElementTree as ET
import json
from dotenv import load_dotenv
load_dotenv()

# 连接到 MySQL 数据库
conn = pymysql.connect(
    host=os.getenv('MYSQL_HOST'),
    user=os.getenv('MYSQL_USERNAME'),
    password=os.getenv('MYSQL_PASSWD'),
    database=os.getenv('MYSQL_DATABASE')
)

def get_select_rows(command):
    #TABLE_STRUCTURE_RECOGNIZE
    # 创建一个游标对象
    cursor = conn.cursor()
    # 执行 SQL 查询
    cursor.execute(command)
    # 获取所有结果
    rows = cursor.fetchall()
    # 处理结果
    for row in rows:
        print(row[0])
        print('\n\n')
    # 关闭游标和连接
    cursor.close()
    return rows

def get_parsed_res(row_recognize_res):
    root = ET.fromstring(row_recognize_res)
    table_structure = [row.get('category') for row in root.findall('row')]
    return table_structure

'''
做dataset：
写成{"内容":"结构标签"}的字典，存成json文件
'''
def get_split_dict(text: list, label: list):
    pattern = r'第.*行内容：'
    text = re.sub(pattern, '', text)
    text_list = text.split('\n\n')
    parse_recognize_res = '<rows>' + parse_xml_output_without_thinking(label,['rows'])['rows'][0] + '</rows>'
    table_structure = get_parsed_res(parse_recognize_res)
    result_dict = dict(zip(text_list, table_structure))
    file_path = 'dataset.json'
    # 使用json.dump()方法将字典写入JSON文件
    with open(file_path, 'w') as json_file:
        json.dump(result_dict, json_file, indent=4)
    return result_dict

if __name__ == "__main__":
    command = "SELECT DISTINCT info, model_output FROM flatten_table_all_model_prompt WHERE custom_model_name = 'TABLE_STRUCTURE_RECOGNIZE' LIMIT 10"
    rows = get_select_rows(command)
    for index, row in iter(rows):
        text = row[0]
        label = row[1]
        result_dict = get_split_dict(text, label)
        print(result_dict)