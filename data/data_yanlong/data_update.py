import torch
from . import train_dataset


def train_data(num):

    a = train_dataset.a[:num]

    yuan_x = torch.FloatTensor([0])
    yuan_y = torch.FloatTensor([0])
    QUDAIN1 = []
    QUDAIN2 = []
    QUDAIN3 = []
    QUDIAN_y = []
    QUDIAN_x = []
    ZUOBIAO = []

    i = 0
    ii= 0

    output_list = torch.split(a, split_size_or_sections=3, dim=1)
    QUDAIN1.append(output_list[0])
    QUDAIN2.append(output_list[1])
    QUDAIN1 = torch.cat(QUDAIN1, dim=0) # 位姿
    QUDAIN2 = torch.cat(QUDAIN2, dim=0)
    output_list1 = torch.split(QUDAIN2, split_size_or_sections=2, dim=1)
    QUDAIN3.append(output_list1[1]) # z
    QUDAIN3 = torch.cat(QUDAIN3, dim=0)
    # print(QUDAIN3.size())

    while i <= 49:
        yuan_y = yuan_y + 50/1000
        qudian_y = torch.FloatTensor([yuan_y])
        QUDIAN_y.append(qudian_y)
        i  += 1
    QUDIAN_y = torch.cat(QUDIAN_y, dim=0)
    QUDIAN_y = QUDIAN_y.unsqueeze(1)
    # print(QUDIAN_y.size())

    while ii <= 59:
        yuan_x = yuan_x + 50/1000
        qudian_x = torch.FloatTensor([yuan_x])
        QUDIAN_x.append(qudian_x)
        ii  += 1
    QUDIAN_x = torch.cat(QUDIAN_x, dim=0)
    QUDIAN_x = QUDIAN_x.unsqueeze(1)
    # print(QUDIAN_x.size())

    for x in QUDIAN_x:
        for y in QUDIAN_y:
            zuobiao = torch.FloatTensor([[x, y]])
            ZUOBIAO.append(zuobiao)
    ZUOBIAO = torch.cat(ZUOBIAO, dim=0) # x、y
    # print(ZUOBIAO, ZUOBIAO.size())

    zuizhong = torch.cat((QUDAIN1, ZUOBIAO, QUDAIN3), dim=1)
    return zuizhong