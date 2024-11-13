import os
import torch

# 使用 os 指定加载文件的路径和文件名
load_dir = '/home/cn/RPSN_3/data/data_cainan'
file_name = 'train_dataset_1000.pt'

# 拼接路径和文件名
file_path = os.path.join(load_dir, file_name)

# 从指定路径加载张量
data_tensor = torch.load(file_path)

# 查看数据
print("加载的张量的形状：", data_tensor.shape)
print("前5个张量数据：")
print(data_tensor[:5])