import numpy as np
import torch
import os
import math
import sys
sys.path.append("..")
from lib.trans_all import shaping
from lib.IK import calculate_IK
from lib.IK_loss import calculate_IK_loss


def data_generate(i):
    data = []
    a_IK = torch.tensor([0, -0.6127, -0.57155, 0, 0, 0])
    d_IK = torch.tensor([0.1807, 0, 0, 0.17415, 0.11985, 0.11655])
    alpha_IK = torch.FloatTensor([math.pi / 2, 0, 0, math.pi / 2, -math.pi / 2, 0])
    for a in range (i):
        if a % 9 == 0:
            data_echo = []

            yuanxin_x, yuanxin_y, yaw_yuanxin = generrate_yuanxin()

            yuanxin = [0, 0, yaw_yuanxin, yuanxin_x, yuanxin_y, 0]
            yuanxin = [round(val_yuanxin, 3) for val_yuanxin in yuanxin]
            yuanxin_tensor = torch.FloatTensor([yuanxin])

            for num_data in range(np.random.randint(1, 8)):

                MLP_output_base = shaping(yuanxin_tensor)
                # 如果当前底盘位置和物品点位IK出现错误无解则随机产生物品点，直到能够解出
                num_incorrect = 1
                while num_incorrect == 1:
                    # 取在移动底盘不同时，同时在机械臂可达范围内的的点 
                    tensor = generrate_dian(yuanxin[3], yuanxin[4])
                    # IK检查
                    IK_test_tensor = torch.FloatTensor([tensor])
                    # 转换为输入IK的旋转矩阵
                    input_tar = shaping(IK_test_tensor).view(4, 4)
                    angle_solution, num_Error1, num_Error2, the_NANLOSS_of_illegal_solution_with_num_and_Nan = calculate_IK(
                                            input_tar, MLP_output_base, a_IK, d_IK, alpha_IK
                        )
                    IK_loss, num_incorrect, num_correct = calculate_IK_loss(
                        angle_solution, the_NANLOSS_of_illegal_solution_with_num_and_Nan
                        )

                data_echo.append(tensor)

            list_0 = [0, 0, 0, 0, 0, 0]
            while  num_data < 6:
                data_echo.append(list_0)
                num_data += 1

            data.append(data_echo)
            # print(data_echo)
            # print("完成一组", a)
        else:
            data_echo = []

            yuanxin_x, yuanxin_y, yaw_yuanxin = generrate_yuanxin()

            yuanxin = [0, 0, yaw_yuanxin, yuanxin_x, yuanxin_y, 0]
            yuanxin = [round(val_yuanxin, 3) for val_yuanxin in yuanxin]
            yuanxin_tensor = torch.FloatTensor([yuanxin])
            iiii = 1
            while iiii<8:

                MLP_output_base = shaping(yuanxin_tensor)
                # 如果当前底盘位置和物品点位IK出现错误无解则随机产生物品点，直到能够解出
                num_incorrect = 1
                while num_incorrect == 1:
                    # 取在移动底盘不同时，同时在机械臂可达范围内的的点 
                    tensor = generrate_dian(yuanxin[3], yuanxin[4])
                    # IK检查
                    IK_test_tensor = torch.FloatTensor([tensor])
                    # 转换为输入IK的旋转矩阵
                    input_tar = shaping(IK_test_tensor).view(4, 4)
                    angle_solution, num_Error1, num_Error2, the_NANLOSS_of_illegal_solution_with_num_and_Nan = calculate_IK(
                                            input_tar, MLP_output_base, a_IK, d_IK, alpha_IK
                        )
                    IK_loss, num_incorrect, num_correct = calculate_IK_loss(
                        angle_solution, the_NANLOSS_of_illegal_solution_with_num_and_Nan
                        )

                iiii += 1
                data_echo.append(tensor)
            # print(data_echo)

            data.append(data_echo)
            # print("完成一组", a)
    
    data_tensor = torch.FloatTensor(data)

    return data, data_tensor

def generrate_yuanxin():
    yuanxin_x = np.random.uniform(-0.4, 4.4)
    yuanxin_y = np.random.uniform(-0.4, 3.0)
    if 0 <= yuanxin_x <= 4:
        while 0 <= yuanxin_y <= 2.6:
            yuanxin_y = np.random.uniform(-0.4, 3.0)
    yaw_yuanxin = np.random.uniform(-np.pi, np.pi)

    return yuanxin_x, yuanxin_y, yaw_yuanxin

def generrate_dian(yuanxin_x, yuanxin_y):

    x = np.random.uniform(0, 4)
    y = np.random.uniform(0, 2.6)
    distance_yuan_and_dian = math.sqrt((x - yuanxin_x)**2 + (y - yuanxin_y)**2)
    while distance_yuan_and_dian >= 1.3:
        x = np.random.uniform(0, 4)
        y = np.random.uniform(0, 2.6)
        distance_yuan_and_dian = math.sqrt((x - yuanxin_x)**2 + (y - yuanxin_y)**2)

    z = np.random.uniform(1, 1.1)

    yaw = np.random.uniform(-np.pi, np.pi)
    pitch = np.random.uniform(-(np.pi)/2, (np.pi)/2)
    roll = np.random.uniform(-np.pi, np.pi)

    tensor = [roll, pitch, yaw, x, y, z]

    tensor = [round(val, 3) for val in tensor] # 保留3位小数

    return tensor


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

    save_dir_train = '/home/cn/RPSN_4/data/data_cainan/1000/train'
    file_name_txt = 'train_dataset_1000.txt'
    file_name_tensor = 'train_dataset_1000.pt'

    data, data_tensor = data_generate(1000)

    save_data(data, save_dir_train, file_name_txt)
    save_data_tensor(data_tensor, save_dir_train, file_name_tensor)

