import os
import pickle

    
def save_base_table_dict(base_table_dict: dict, file_name: str):
    save_path = r"./data/save_files"
    os.makedirs(save_path, exist_ok=True)  # 如果文件夹不存在，则创建它，避免报错
    full_path = os.path.join(save_path, file_name)
    with open(full_path, 'wb') as file:
        pickle.dump(base_table_dict, file)
    print(f"base_table_dict 已保存为 '{full_path}'。")

def load_base_table_dict(file_path: str):
    with open(file_path, 'rb') as file:
        return pickle.load(file)