import pickle
import os
from tqdm import tqdm

# 设置数据目录的路径
data_directory = 'data/'  # 替换为你的数据目录的实际路径
dump_directory = '../XGaussian/Xray_data/'

# 列出所有的 .pickle 文件
pickle_files = [f for f in os.listdir(data_directory) if f.endswith('.pickle')]

# 循环处理每个文件
for file_name in tqdm(pickle_files):
    path_to_file = os.path.join(data_directory, file_name)
    path_to_dump = os.path.join(dump_directory, file_name)
    
    # 加载每个 pickle 文件并使用协议版本 4 重新保存
    with open(path_to_file, 'rb') as handle:
        data = pickle.load(handle)
    
    with open(path_to_dump, 'wb') as handle:
        pickle.dump(data, handle, protocol=4)

print("所有文件处理完成。")


