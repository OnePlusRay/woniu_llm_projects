import pickle
import os
import random
import time
import datetime


def save_checkpoint(data, checkpoint, data_file_name, checkpoint_file_name):
    '''保存检查点和数据'''
    with open(checkpoint_file_name, 'wb') as f:
        pickle.dump(checkpoint, f)
    save_base_table_dict(data, data_file_name)


def load_checkpoint(data_file_path, checkpoint_file_path):
    '''加载检查点和数据'''
    if os.path.exists(checkpoint_file_path):
        with open(checkpoint_file_path, 'rb') as f:
            checkpoint = pickle.load(f)
    data = load_base_table_dict(data_file_path)
    return checkpoint, data


def save_base_table_dict(base_table_dict, file_name):
    '''保存变量base_table_dict'''
    with open(file_name, 'wb') as file:
        pickle.dump(base_table_dict, file)
    print(f"base_table_dict 已成功保存为 '{file_name}'")
    

def load_base_table_dict(file_path):
    '''加载变量base_table_dict'''
    with open(file_path, 'rb') as file:
        return pickle.load(file)

    
# 把中间处理excel的过程放进来
# def checkpoint(start, end, checkpoint_interval=5):
#     '''主函数：保存检查点和中间变量，每次调用从上一个检查点（若有）开始运行'''
#     # 尝试加载检查点
#     checkpoint = load_checkpoint()
#     if checkpoint:
#         start = checkpoint['current']
#         print(f"Resuming from checkpoint: {start}")

#     for i in range(start, end):
#         # 模拟长时间运行的任务
#         print(f"Processing {i}")
        
#         # 模拟报错
#         if random.random() < 0.1:  # 10%的概率触发错误
#             save_checkpoint({'current': i})  # 在报错前保存检查点
#             raise Exception(f"Simulated error at step {i}")
        
#         # 每隔 checkpoint_interval 次保存一次检查点
#         if i % checkpoint_interval == 0:
#             save_checkpoint({'current': i})
        
#         # 模拟处理时间
#         time.sleep(1)

#     # 任务完成后删除检查点文件
#     if os.path.exists(checkpoint_file):
#         os.remove(checkpoint_file)
#     print("Task completed")

# if __name__ == "__main__":
    # try:
    #     checkpoint(0, 20)
    # except KeyboardInterrupt:
    #     print("Task interrupted")
    # except Exception as e:
    #     print(f"Task failed with error: {e}")
    #     print("You can restart the program to continue from the last checkpoint.")
