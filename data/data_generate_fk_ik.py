import numpy as np
import torch
import os
import math
import sys
sys.path.append("..")
from lib.trans_all import shaping, rot2euler
from lib.IK import calculate_IK
from lib.IK_loss import calculate_IK_loss

from lib.FK import get_zong_t


def data_generate(i):
    data = []
    data_dipan = []
    a_IK = torch.tensor([0, -0.6127, -0.57155, 0, 0, 0])
    d_IK = torch.tensor([0.1807, 0, 0, 0.17415, 0.11985, 0.11655])
    alpha_IK = torch.FloatTensor([math.pi / 2, 0, 0, math.pi / 2, -math.pi / 2, 0])
    for a in range (i):
        data_echo = []
        data_dipan_echo = []
        while not len(data_echo)==7:
            yuanxin_x, yuanxin_y, yaw_yuanxin = generrate_yuanxin()
            yuanxin = [0, 0, yaw_yuanxin, yuanxin_x, yuanxin_y, 0]
            # yuanxin = [round(val_yuanxin, 3) for val_yuanxin in yuanxin]
            yuanxin_tensor = torch.FloatTensor([yuanxin])
            MLP_output_base = shaping(yuanxin_tensor)

            for num_data in range(np.random.randint(1, 8)):

                # tensor = generrate_dian_fk(a_IK, d_IK, alpha_IK, yuanxin_x, yuanxin_y)
                # data_echo.append(tensor)
                num_incorrect = 1
                # num_gegeg = 0
                while num_incorrect == 1:
                    tensor = generrate_dian_fk(a_IK, d_IK, alpha_IK, yuanxin_x, yuanxin_y)
                    IK_test_tensor = torch.FloatTensor([tensor])
                    # 转换为输入IK的旋转矩阵
                    input_tar = shaping(IK_test_tensor).view(4, 4)
                    angle_solution, num_Error1, num_Error2, the_NANLOSS_of_illegal_solution_with_num_and_Nan = calculate_IK(
                                            input_tar, MLP_output_base, a_IK, d_IK, alpha_IK
                        )
                    IK_loss, num_incorrect, num_correct = calculate_IK_loss(
                        angle_solution, the_NANLOSS_of_illegal_solution_with_num_and_Nan
                        )
                    # num_gegeg += 1
                    # print("不是吧哥们", num_gegeg)
                data_echo.append(tensor)

            list_0 = [0, 0, 0, 0, 0, 0]
            while  num_data < 6:
                data_echo.append(list_0)
                num_data += 1

        data.append(data_echo)
        data_dipan.append(yuanxin)

        print("完成一组", a)
    data_tensor = torch.FloatTensor(data)
    data_dipan_tensor = torch.FloatTensor(data_dipan)

    return data, data_tensor, data_dipan_tensor

def generrate_yuanxin():
    yuanxin_x = np.random.uniform(-0.4, 4.4)
    yuanxin_y = np.random.uniform(-0.4, 3.0)
    if 0 <= yuanxin_x <= 4:
        while 0 <= yuanxin_y <= 2.6:
            yuanxin_y = np.random.uniform(-0.4, 3.0)
    yaw_yuanxin = np.random.uniform(-np.pi, np.pi)

    return yuanxin_x, yuanxin_y, yaw_yuanxin


def generrate_dian_fk(a_IK, d_IK, alpha_IK, yuanxin_x, yuanxin_y):

    theta = [0, 0, 0, 0, 0, 0]
    for i in range(6):
        theta[i] = np.random.uniform(-np.pi, np.pi)

    TT = get_zong_t(a_IK, d_IK, alpha_IK, theta)
    px = TT[0, 3] + yuanxin_x
    py = TT[1, 3] + yuanxin_y
    pz = TT[2, 3]

    while not (0<px<4 and 0<py<2.6 and 0<pz<0.1):
        for i in range(6):
            theta[i] = np.random.uniform(-np.pi, np.pi)

        TT = get_zong_t(a_IK, d_IK, alpha_IK, theta)
        px = TT[0, 3] + yuanxin_x
        py = TT[1, 3] + yuanxin_y
        pz = TT[2, 3]

    nx = TT[0, 0]
    ny = TT[1, 0]
    nz = TT[2, 0]
    ox = TT[0, 1]
    oy = TT[1, 1]
    oz = TT[2, 1]
    ax = TT[0, 2]
    ay = TT[1, 2]
    az = TT[2, 2]  

    rot = np.array([
        [nx, ox, ax],
        [ny, oy, ay],
        [nz, oz, az]
    ])

    euler = rot2euler(rot)
    roll = euler[0]
    pitch = euler[1]
    yaw = euler[2]

    tensor = [roll, pitch, yaw, px, py, pz]
    # tensor = [round(val, 3) for val in tensor] # 保留3位小数

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

def save_MLP_output(data_complite, save_dir, file_name):

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    file_path = os.path.join(save_dir, file_name)

    with open(file_path, 'w') as f:
        for tensor_str in data_complite:
            
            tensor_str = ' '.join(map(str, tensor_str))  # 将 tensor 转换为字符串并用空格分隔
            f.write(tensor_str + '\n')
        f.write('\n')

def save_data_tensor(data_tensor, save_dir, file_name_tensor):

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    file_path = os.path.join(save_dir, file_name_tensor)

    torch.save(data_tensor, file_path)

if __name__ == "__main__":

    save_dir_train = '/home/cn/RPSN_4/data/data_cainan/5000-fk-all-random-with-dipan/train'
    file_name_txt = 'train_dataset_5000.txt'
    file_name_tensor = 'train_dataset_5000.pt'
    file_name_dipan_tensor = "train_dataset_dipan_5000.pt"

    data, data_tensor, data_dipan_tensor = data_generate(5000)

    save_data(data, save_dir_train, file_name_txt)
    save_data_tensor(data_tensor, save_dir_train, file_name_tensor)
    save_data_tensor(data_dipan_tensor, save_dir_train, file_name_dipan_tensor)
