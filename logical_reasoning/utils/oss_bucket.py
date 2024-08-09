import os
import oss2
import requests

def get_oss_bucket():
    auth = oss2.Auth(os.getenv("ACCESS_KEY_ID"), os.getenv("ACCESS_KEY_SECRET"))
    bucket = oss2.Bucket(auth, os.getenv('ENDPONIT'), os.getenv('OSS_BUCKET_NAME'))
    return bucket


oss_bucket = get_oss_bucket()

#检查url中文件是否存在
def check_file_exists(url):
    response = requests.head(url)
    if response.status_code == 200:
        print('文件已存在')
        return True
    else:
        print('文件不存在')
        return False

#获取文件url，如url中文件不存在，上传至阿里云
def get_bucket_url(local_path, type: str, oss_filename):
    bucket_path = rf"flatten_table/data/{type}/{oss_filename}"
    bucket_url = f"https://{os.getenv('OSS_BUCKET_NAME')}.{os.getenv('ENDPONIT')}/{bucket_path}"
    if not check_file_exists(bucket_url):
        oss_bucket.put_object_from_file(bucket_path, local_path)
    return bucket_url
    

