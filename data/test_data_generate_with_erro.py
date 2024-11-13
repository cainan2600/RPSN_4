import numpy as np
import torch
import os
import math


def data_generate(i):
    data = []
    for a in range (i):
        # 取四分之一的错误集合
        if a/4 == 0:
            data_echo = []
            for num_data in range(np.random.randint(1, 6)):

                yuanxin_x = np.random.uniform(-0.4, 4.4)
                yuanxin_y = np.random.uniform(-0.4, 3.0)
                if 0 <= yuanxin_x <= 4:
                    while 0 <= yuanxin_y <= 2.6:
                        yuanxin_y = np.random.uniform(-0.4, 3.0)
                # print(yuanxin_x, yuanxin_y)

                # 取在移动底盘不同时，同时在机械臂可达范围内的的点
                x = np.random.uniform(0, 4)
                y = np.random.uniform(0, 2.6)
                distance_yuan_and_dian = math.sqrt((x - yuanxin_x)**2 + (y - yuanxin_y)**2)

                # 取求解不出的点------一半在可达范围外
                if not num_data/2 == 0:
                    while distance_yuan_and_dian >= 1.3:
                        x = np.random.uniform(0, 4)
                        y = np.random.uniform(0, 2.6)
                        distance_yuan_and_dian = math.sqrt((x - yuanxin_x)**2 + (y - yuanxin_y)**2)
                else:
                    while distance_yuan_and_dian <= 1.3:
                        x = np.random.uniform(0, 4)
                        y = np.random.uniform(0, 2.6)
                        distance_yuan_and_dian = math.sqrt((x - yuanxin_x)**2 + (y - yuanxin_y)**2)                

                z = np.random.uniform(1, 1.1)

                yaw = np.random.uniform(-np.pi, np.pi)
                pitch = np.random.uniform(-(np.pi)/2, (np.pi)/2)
                roll = np.random.uniform(-np.pi, np.pi)

                tensor = [roll, pitch, yaw, x, y, z]

                tensor = [round(val, 3) for val in tensor] # 保留3位小数

                data_echo.append(tensor)

            list_0 = [0, 0, 0, 0, 0, 0]
            while  num_data < 6:
                data_echo.append(list_0)
                num_data += 1

            data.append(data_echo)
        else:
            data_echo = []
            for num_data in range(np.random.randint(1, 8)):

                yuanxin_x = np.random.uniform(-0.4, 4.4)
                yuanxin_y = np.random.uniform(-0.4, 3.0)
                if 0 <= yuanxin_x <= 4:
                    while 0 <= yuanxin_y <= 2.6:
                        yuanxin_y = np.random.uniform(-0.4, 3.0)
                # print(yuanxin_x, yuanxin_y)

                # 取在移动底盘不同时，同时在机械臂可达范围内的的点
                x = np.random.uniform(0, 4)
                y = np.random.uniform(0, 2.6)
                distance_yuan_and_dian = math.sqrt((x - yuanxin_x)**2 + (y - yuanxin_y)**2)

                # 取求解不出的点------一半在可达范围外
                # if not num_data/2 == 0:
                while distance_yuan_and_dian >= 1.3:
                    x = np.random.uniform(0, 4)
                    y = np.random.uniform(0, 2.6)
                    distance_yuan_and_dian = math.sqrt((x - yuanxin_x)**2 + (y - yuanxin_y)**2)
                # else:
                #     if distance_yuan_and_dian <= 1.3:
                #         x = np.random.uniform(0, 2.6)
                #         y = np.random.uniform(0, 4)
                #         distance_yuan_and_dian = math.sqrt((x - yuanxin_x)**2 + (y - yuanxin_y)**2)                

                z = np.random.uniform(1, 1.1)

                yaw = np.random.uniform(-np.pi, np.pi)
                pitch = np.random.uniform(-(np.pi)/2, (np.pi)/2)
                roll = np.random.uniform(-np.pi, np.pi)

                tensor = [roll, pitch, yaw, x, y, z]

                tensor = [round(val, 3) for val in tensor] # 保留3位小数

                data_echo.append(tensor)

            list_0 = [0, 0, 0, 0, 0, 0]
            while  num_data < 6:
                data_echo.append(list_0)
                num_data += 1

            data.append(data_echo)
    
    data_tensor = torch.FloatTensor(data)

    return data, data_tensor

def save_data(data_complite, save_dir, file_name):

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    file_path = os.path.join(save_dir, file_name)

    with open(file_path, 'w') as f:
        for tensor in data_complite:
            for tensor_1 in tensor:
                tensor_str = ' '.join(map(str, tensor_1))  # 将 tensor 转换为字符串并用空格分隔
                f.write(tensor_str + '\n')
            f.write('\n')

def save_data_tensor(data_tensor, save_dir, file_name_tensor):

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    file_path = os.path.join(save_dir, file_name_tensor)

    torch.save(data_tensor, file_path)

if __name__ == "__main__":

    save_dir_test = '/home/cn/RPSN_3/data/data_cainan/5000/test_with_erro'
    file_name_txt = 'test_dataset_400_with_erro.txt'
    file_name_tensor = 'test_dataset_400_with_erro.pt'

    data, data_tensor = data_generate(400)

    save_data(data, save_dir_test, file_name_txt)
    save_data_tensor(data_tensor, save_dir_test, file_name_tensor)

