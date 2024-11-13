import torch
import math

# def find_closest(angle_solution, where_is_the_illegal):

#     the_NANLOSS_of_illegal_solution_with_num_and_Nan = 0

#     min_distance = 100  # 记录非法数据中，距离3.14最近的数的绝对值距离，初始化为一个足够大的值
#     min_index = []      # 记录比较后距离3.14最近的值的索引
#     single_ik_loss = torch.tensor(0.0, requires_grad=True)

#     for index in where_is_the_illegal:
#         there_exist_nan = 0
#         i, j = index
#         if math.isnan(angle_solution[i][j]):
#             pass
#         else:
#             for angle in range(6):
#                 if math.isnan(angle_solution[i][angle]):
#                     there_exist_nan +=1
#             if there_exist_nan == 0:
#                 num = angle_solution[i][j]
#                 distance = abs(num) - (torch.pi)          # 计算拿出来的值距离(pi)的距离
#                 if distance < min_distance:
#                     min_distance = distance
#                     min_index = index
#             else:
#                 pass
#         single_ik_loss = single_ik_loss + min_distance
#     return the_NANLOSS_of_illegal_solution_with_num_and_Nan

def find_closest(angle_solution, where_is_the_illegal):
    min_distance = 100  # 记录非法数据中，距离3.14最近的数的绝对值距离，初始化为一个足够大的值
    min_index = []      # 记录比较后距离3.14最近的值的索引

    single_ik_loss = torch.tensor(0.0, requires_grad=True)

    the_NANLOSS_of_illegal_solution_with_num_and_Nan = 0
    # print(' angle_solution', angle_solution)
    # print(' where_is_the_illegal',  where_is_the_illegal)
    # print('save_what_caused_Error2_as_Nan',save_what_caused_Error2_as_Nan)

    for index in where_is_the_illegal:
        there_exist_nan = 0
        i, j = index
        if math.isnan(angle_solution[i][j]):
            pass
            # single_ik_loss = single_ik_loss + (abs(save_what_caused_Error2_as_Nan[i])-torch.tensor([1]))*1000
            # print(single_ik_loss)
        else:
            for angle in range(6):
                if math.isnan(angle_solution[i][angle]):
                    there_exist_nan +=1
            if there_exist_nan == 0:
                # print(angle_solution[i][j])
                num = angle_solution[i][j]
                distance = abs(num) - (torch.pi)          # 计算拿出来的值距离(pi)的距离
                single_ik_loss = single_ik_loss + distance
                # print(single_ik_loss)
                if distance < min_distance:
                    min_distance = distance
                    min_index = index
            else:
                pass
                # single_ik_loss = single_ik_loss + (abs(save_what_caused_Error2_as_Nan[i]) - torch.tensor([1])) * 1000
                # print(single_ik_loss)
        single_ik_loss = single_ik_loss + min_distance
    return (single_ik_loss + the_NANLOSS_of_illegal_solution_with_num_and_Nan)
    # return the_NANLOSS_of_illegal_solution_with_num_and_Nan