import torch
import math 
import numpy as np

# 定义可微分形式
def cos(a):
    return torch.cos(a)


# 定义可微分形式
def sin(a):
    return torch.sin(a)



# 用于逆运算的 转置t
def transpose(x): # 取出每个——batch_size中的一个数据集的一个参数

    x = x[0]

    a = x[0][:3]
    b = x[1][:3]
    c = x[2][:3]
    result = torch.stack([a, b, c], 0)

    d = x[0][3]
    e = x[1][3]
    f = x[2][3]
    D = torch.stack([d, e, f], dim=0)
    D = D.unsqueeze(1)

    result_trans = torch.t(result)
    result_mul = torch.mm(-result_trans, D)

    T_Transpose0 = torch.cat([torch.t(result_trans), torch.t(result_mul)], 0)

    P = torch.tensor([0, 0, 0, 1])
    P = P.unsqueeze(0)

    T_Transpose = torch.cat([torch.t(T_Transpose0), P], 0)


    return T_Transpose


# Atan2函数参与BP过程定义
class Atan2Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, y, x):
        result = math.atan2(y, x)
        ctx.save_for_backward(x, y)
        return torch.tensor(result, requires_grad=True)

    @staticmethod
    def backward(ctx, grad_output):
        x, y = ctx.saved_tensors
        grad_y = x / (x ** 2 + y ** 2)
        grad_x = -y / (x ** 2 + y ** 2)
        return grad_output * grad_y, grad_output * grad_x
atan2 = Atan2Function.apply

# 输入三个欧拉角，以tensor形式运算出3×3旋转矩阵
def euler_to_rotMat(yaw, pitch, roll):
    ffff = torch.tensor(0)

    gggg = torch.tensor(1)

    Rz_yaw0 = torch.stack([torch.cos(yaw), -torch.sin(yaw), ffff], 0)
    Rz_yaw1 = torch.stack([torch.sin(yaw), torch.cos(yaw), ffff], 0)
    Rz_yaw2 = torch.stack([ffff, ffff, gggg], 0)
    Rz_yaw = torch.stack([Rz_yaw0, Rz_yaw1, Rz_yaw2], 0)

    Ry_pitch0 = torch.stack([torch.cos(pitch), ffff, torch.sin(pitch)], 0)
    Ry_pitch1 = torch.stack([ffff, gggg, ffff], 0)
    Ry_pitch2 = torch.stack([-torch.sin(pitch), ffff, torch.cos(pitch)], 0)
    Ry_pitch = torch.stack([Ry_pitch0, Ry_pitch1, Ry_pitch2], 0)

    Rx_roll0 = torch.stack([gggg, ffff, ffff], 0)
    Rx_roll1 = torch.stack([ffff, torch.cos(roll), -torch.sin(roll)], 0)
    Rx_roll2 = torch.stack([ffff, torch.sin(roll), torch.cos(roll)], 0)
    Rx_roll = torch.stack([Rx_roll0, Rx_roll1, Rx_roll2], 0)

    rotMat = torch.mm(Rz_yaw, torch.mm(Ry_pitch, Rx_roll))
    return rotMat


# # 输入1×6tensor形式数据，数据前3个是欧拉角（转换为旋转矩阵），后三个是位置，输出是shaping后的4×4tensor齐次矩阵
# def shaping(x):
    
#     inputs_list_1x6 = shaping_inputs_12to6(x)
#     # print(inputs_list_1x6[0])

#     for j in range(2):
#         x = inputs_list_1x6[j]
#         T_shapings1 = []
#         T_shapings2 = []

#         for i in x:
#             a = i[0]
#             b = i[1]
#             c = i[2]
#             result = euler_to_rotMat(c, b, a) # 得到3x3的旋转矩阵

#             d = i[3]
#             e = i[4]
#             f = i[5]

#             D = torch.stack([d, e, f], dim=0)
#             D = D.unsqueeze(1) # 1x3变为3x1

#             T_shaping0 = torch.cat([torch.t(result), torch.t(D)], 0)
#             P = torch.tensor([0.0, 0.0, 0.0, 1.0])
#             P = P.unsqueeze(0)

#             # T_shaping = torch.cat([torch.t(T_shaping0), P], 0)
#             # T_shaping = T_shaping.unsqueeze(0)

#             if j == 0:
#                 T_shaping1 = torch.cat([torch.t(T_shaping0), P], 0)
#                 T_shaping1 = T_shaping1.unsqueeze(0)
#                 T_shapings1.append(T_shaping1)
#                 # T_shapings1 = torch.tensor([item.cpu().detach().numpy() for item in T_shapings1])
#                 # T_shapings1 = torch.cat(T_shapings1, dim=0)
#             else:
#                 T_shaping2 = torch.cat([torch.t(T_shaping0), P], 0)
#                 T_shaping2 = T_shaping2.unsqueeze(0)
#                 T_shapings2.append(T_shaping2)
#                 # T_shapings2 = torch.tensor([item.cpu().detach().numpy() for item in T_shapings2])
#         T_shapings1 = torch.cat(T_shapings1, dim=0)   
#         T_shapings2 = torch.cat(T_shapings2, dim=0)

#     # T_shapings = torch.cat
#     return T_shapings


# 输入1×6tensor形式数据，数据前3个是欧拉角（转换为旋转矩阵），后三个是位置，输出是shaping后的4×4tensor齐次矩阵
def shaping(x):
    T_shapings = []
    for i in x: # 取出每个——batch_size中的一个数据集的一个参数
        a = i[0]
        b = i[1]
        c = i[2]
        result = euler_to_rotMat(c, b, a)
        # print(result)

        d = i[3]
        e = i[4]
        f = i[5]

        D = torch.stack([d, e, f], dim=0)
        D = D.unsqueeze(1) # 1x3变为3x1

        T_shaping0 = torch.cat([torch.t(result), torch.t(D)], 0)
        P = torch.tensor([0.0, 0.0, 0.0, 1.0])
        P = P.unsqueeze(0)

        T_shaping = torch.cat([torch.t(T_shaping0), P], 0)
        T_shaping = T_shaping.unsqueeze(0)
        T_shapings.append(T_shaping)

    T_shapings = torch.cat(T_shapings, dim=0)
    # print(T_shapings.size())
    return T_shapings


def shaping2(x):
    T_shapings = []

    for i in x:
        a = torch.tensor(0, requires_grad=False)
        b = torch.tensor(0, requires_grad=False)
        c = i[2]
        result = euler_to_rotMat(c, b, a)

        d = i[3]
        e = i[4]
        f = torch.tensor(0, requires_grad=False)

        D = torch.stack([d, e, f], dim=0)
        D = D.unsqueeze(1)

        T_shaping0 = torch.cat([torch.t(result), torch.t(D)], 0)
        P = torch.tensor([0.0, 0.0, 0.0, 1.0])
        P = P.unsqueeze(0)

        T_shaping = torch.cat([torch.t(T_shaping0), P], 0)
        T_shaping = T_shaping.unsqueeze(0)
        T_shapings.append(T_shaping)

    T_shapings = torch.cat(T_shapings, dim=0)
    return T_shapings

def shaping_inputs_6to12(ori_position, tar_object_position):
    # 将目标物体1x6与放置位置1x6组合为1x12
    inputs_list_1x12 = []

    for position_tar in tar_object_position:

        new_list = torch.cat((position_tar, ori_position))
        inputs_list_1x12.append(new_list)
    
    inputs_list_1x12 = torch.tensor([np.array(item) for item in inputs_list_1x12])


    return inputs_list_1x12

def shaping_inputs_12to6(inputs_list_1x12):
    # 将1x12输入转为10x1x6,
    inputs_list_1x6 = []

    inputs_list = torch.split(inputs_list_1x12, split_size_or_sections=6, dim=0)

    for input in inputs_list:
        inputs_list_1x6.append(input)
    inputs_list_1x6 = torch.cat(inputs_list_1x6, dim=0)

    return inputs_list_1x6

def shaping_output_6to3(intermediate_outputs):
    # 5x6转换为10x3
    
    outputs_list_1x3 = []

    output_list = torch.split(intermediate_outputs, split_size_or_sections=3, dim=1)

    for output in output_list:
        outputs_list_1x3.append(output)
    outputs_list_1x3 = torch.cat(outputs_list_1x3, dim=0)

    return outputs_list_1x3

def shaping_inputs_xx6_to_1xx(inputs_xx6):

    inputs_1xx = []
    for inputs_1x6 in inputs_xx6:
        inputs_1xx.append(inputs_1x6)
    inputs_1xx = torch.cat(inputs_1xx, dim=0)

    return inputs_1xx

def shaping_inputs_1xx_to_xx1x6(inputs_1xx, num_i):
    # 将1x42转换为7x1x6
    inputs_xx1x6 = []
    h = int(num_i / 6)
    inputs_xx1x6 = inputs_1xx.view(h, 1, 6)

    return inputs_xx1x6

def shaping_outputs_1xx_to_xx3(intermediate_outputs, num_i):
    # 将1x42转换为7x1x6

    h = int(num_i / 6)
    outputs_xx3 = intermediate_outputs.view(h, 3)

    return outputs_xx3