import torch
import math
from lib.find_closest import find_closest
import numpy as np


inputs_of_final_result = []
outputs_of_MLP = []
final_result = []

# angle_solution传入ik运算的8组解或异常跳出的值，loss由此函数定义部分（总loss还有其他两部分）
def calculate_IK_loss(angle_solution, the_NANLOSS_of_illegal_solution_with_num_and_Nan):

    num_incorrect = 0
    num_correct = 0
    num_illegal = 0
    IK_loss = torch.tensor([0.0], requires_grad=True)
    legal_solution = []
    where_is_the_illegal = []
    fanwei = [np.pi * 178/180, np.pi * 130/180, np.pi * 135/180, np.pi * 178/180, np.pi * 128/180, np.pi]
    if len(angle_solution) == 1:  # 判断是不是IK异常跳出的，如果是直接赋值给loss
        num_incorrect += 1
        IK_loss = IK_loss + angle_solution

    else:
        # 不报错的IK运算有8组解，每组解6个关节值，这里的关节值可能是NaN
        for solution_index in range(8):
            ls = []
            for angle_index in range(6):
                if -fanwei[angle_index] <= angle_solution[solution_index][angle_index] <= fanwei[angle_index]:
                    ls.append(angle_solution[solution_index][angle_index])
                else:
                    num_illegal += 1
                    where_is_the_illegal.append([solution_index, angle_index])                  
                    break

            if len(ls) == 6:
                num_correct += 1
                legal_solution.append(ls)
                IK_loss = IK_loss + torch.tensor([0])
                break

        if num_illegal == 8:
            IK_loss = IK_loss + find_closest(angle_solution, where_is_the_illegal, the_NANLOSS_of_illegal_solution_with_num_and_Nan) #!!!!!优先惩罚nan产生项，loss定义在计算过程中

            num_incorrect += 1

    return IK_loss, num_incorrect, num_correct

def calculate_IK_loss_test(angle_solution):

    IK_loss_test_incorrect = 0
    IK_loss_test_correct = 0

    num_illegal = 0
    IK_loss = torch.tensor([0.0], requires_grad=True)
    legal_solution = []
    where_is_the_illegal = []
    fanwei = [np.pi * 178/180, np.pi * 130/180, np.pi * 135/180, np.pi * 178/180, np.pi * 128/180, np.pi]

    if len(angle_solution) == 1:  # 判断是不是IK异常跳出的，如果是直接赋值给loss
        IK_loss_test_incorrect += 1
        IK_loss = IK_loss + angle_solution

    else:
        # 不报错的IK运算有8组解，每组解6个关节值，这里的关节值可能是NaN
        for solution_index in range(8):
            ls = []
            for angle_index in range(6):
                if -fanwei[angle_index] <= angle_solution[solution_index][angle_index] <= fanwei[angle_index]:
                    ls.append(angle_solution[solution_index][angle_index])
                else:
                    num_illegal += 1
                    where_is_the_illegal.append([solution_index, angle_index])                  
                    break
            if len(ls) == 6:
                legal_solution.append(ls)

                IK_loss_test_correct += 1

                # inputs_of_final_result.append(aaaaaaaaaa)
                # outputs_of_MLP.append(bbbbbbbbbb)
                final_result.append(ls)
                IK_loss = IK_loss + torch.tensor([0])
                break

        if num_illegal == 8:
            IK_loss = IK_loss + find_closest(angle_solution, where_is_the_illegal, the_NANLOSS_of_illegal_solution_with_num_and_Nan=torch.tensor(0))

            IK_loss_test_incorrect += 1

    # print(IK_loss)
    return IK_loss, IK_loss_test_incorrect, IK_loss_test_correct