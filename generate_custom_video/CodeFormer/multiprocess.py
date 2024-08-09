import os
import shutil
import time
import subprocess

def split_folder_into_subfolders(source_folder, target_folder, subfolder_count):
    files = [f for f in os.listdir(source_folder) if os.path.isfile(os.path.join(source_folder, f))]
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    files_per_folder = len(files) // subfolder_count
    remainder = len(files) % subfolder_count

    start_index = 0
    for i in range(subfolder_count):
        subfolder_path = os.path.join(target_folder, f'subfolder_{i+1}')
        if not os.path.exists(subfolder_path):
            os.makedirs(subfolder_path)
        
        # Calculate the number of files to move to this subfolder
        end_index = start_index + files_per_folder + (1 if i < remainder else 0)
        for file in files[start_index:end_index]:
            shutil.move(os.path.join(source_folder, file), subfolder_path)
        start_index = end_index

def main(base_folder, output_path):
    subfolders = [os.path.join(base_folder, f) for f in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, f))]
    processes = []

    gpu_count = 1  # GPU数量
    processes_per_gpu = 20  # 每个GPU的进程数

    current_gpu = 0
    for index, subfolder in enumerate(subfolders):
        # 分配GPU，循环方式
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(current_gpu)
        proc = subprocess.Popen(['python', 'inference_codeformer.py', '-w', '0.7', '--input_path', subfolder, '--output_path', output_path], env=env)
        processes.append(proc)

        # 更新GPU索引
        if (index + 1) % processes_per_gpu == 0:
            current_gpu = (current_gpu + 1) % gpu_count

    for proc in processes:
        proc.wait()  # Wait for each process to complete

if __name__ == "__main__":
    # Example usage
    base_folder = '/data/disk4/home/xiaohan/xiaohantmp2/test_images'
    output_path = '/data/disk4/home/xiaohan/xiaohantmp2/results'
    split_folder_into_subfolders(base_folder, base_folder, 20)
    start_time = time.time()
    main(base_folder)
    print(f"Model execution and image processing time: {time.time() - start_time:.4f} seconds")