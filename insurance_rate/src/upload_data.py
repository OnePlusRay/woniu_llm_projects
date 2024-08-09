import hashlib
import os
from utils.oss_bucket import get_bucket_url

#获取excel文件hash值（只与文件内容有关），同样内容不同文件名会得到同一个hash值
def get_hash_value(file_path):
    sha256 = hashlib.sha256()
    
    # 以二进制模式读取文件
    with open(file_path, 'rb') as f:
        # 分块读取文件并更新哈希对象
        for chunk in iter(lambda: f.read(4096), b''):
            sha256.update(chunk)
    
    # 返回文件的 SHA-256 哈希值
    hash_value = sha256.hexdigest()
    return hash_value

#上传原始excel文件
def upload_origin_file(file_path):
    hash_value = get_hash_value(file_path)
    print(f'SHA-256: {hash_value}')
    url =  get_bucket_url(file_path, 'origin', f'{hash_value}.xlsx')
    return url

#上传平铺结果Excel文件
def upload_result_file(file_path):
    file_name = os.path.basename(file_path)
    url =  get_bucket_url(file_path, 'result', f'{file_name}')  # 这里可以不加.xlsx
    return url