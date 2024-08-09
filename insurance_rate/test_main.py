from dotenv import load_dotenv
load_dotenv()
from src.get_res import get_res

import os

def get_all_file_paths(directory):
    file_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            file_paths.append(os.path.join(root, file))
    return file_paths



if __name__ == "__main__":
    file_path = r"/sdata/chenrui/ai-project/insurance_rate/data/real_in/中荷样本_小小.xlsx"  # 这个是绝对路径
    output_dir_path = r"./data/real_out"  # 这个是相对路径，注意它是要传到src的get_res文件中，因此需要"./"
    is_test = False  # 是否为小样本测试

    origin_url, result_url = get_res(file_path, output_dir_path, is_test)
    print(origin_url)
    print(result_url)
    # #get_res(file_path, output_dir_path)
    # directory = r'./data/single'
    # all_file_paths = get_all_file_paths(directory)
    # for path in all_file_paths:
    #     original_url, result_url = get_res(path, output_dir_path)
