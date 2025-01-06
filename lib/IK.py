import torch
from lib.trans_all import *


# numError1 = []
# numError2 = []
# numNOError1 = []
# numNOError2 = []
# num_correct_test = []
# num_incorrect_test = []
# numPositionloss_pass = []
# numeulerloss_pass = []




grads = {}
def save_grad(name):
    def hook(grad):
        grads[name] = grad
    return hook

# 输入两个4×4tensor（世界坐标系下目标位置、世界坐标系下底盘位置）
def calculate_IK(input_tar, MLP_output_base, a, d, alpha):

    num_Error1 = 0
    num_Error2 = 0

    save_what_caused_Error2_as_Nan = []
    the_NANLOSS_of_illegal_solution_with_num_and_Nan = torch.tensor([0.0], requires_grad=True)

    TT = torch.mm(transpose(MLP_output_base), input_tar)
    nx = TT[0, 0]
    ny = TT[1, 0]
    nz = TT[2, 0]
    ox = TT[0, 1]
    oy = TT[1, 1]
    oz = TT[2, 1]
    ax = TT[0, 2]
    ay = TT[1, 2]
    az = TT[2, 2]
    px = TT[0, 3]
    py = TT[1, 3]
    pz = TT[2, 3]
    # print(TT, MLP_output_base)

    # 求角1
    m = py - ay*d[5]
    n = px - ax*d[5]
    theta11 = atan2(m, n)
    t1 = torch.stack([theta11, theta11, theta11, theta11, theta11, theta11, theta11, theta11], 0)

    # 求角6
    # AA = (nx**2 + ny**2 + nz**2 - 1) * 1e+7
    # BB = (2*nx*ox + 2*ny*oy + 2*oz*nz) * 1e+7
    # CC = (ox**2 + oy**2 + oz**2 - 1) * 1e+7
    AA = nx**2 + ny**2 + nz**2 - 1
    BB = 2*nx*ox + 2*ny*oy + 2*oz*nz
    CC = ox**2 + oy**2 + oz**2 - 1
    if AA == 0:
        if BB == 0:
            num_Error1 += 1
            angle_solution = torch.tensor([100.0], requires_grad=True)

            return angle_solution, num_Error1, num_Error2, the_NANLOSS_of_illegal_solution_with_num_and_Nan
        else:
            theta61 = atan2(-CC, BB)
            t6 = torch.stack([theta61, theta61, theta61, theta61, theta61, theta61, theta61, theta61], 0)
            # t6.register_hook(save_grad('t6'))
            # print("[grads]t6:", grads)
            # print("角61推出来了", t6)

    else:

        if BB**2 - 4*AA*CC > 0:
            theta61 = atan2(-BB + torch.sqrt(BB**2 - 4*AA*CC), 2*AA)
            theta62 = atan2(-BB - torch.sqrt(BB**2 - 4*AA*CC), 2*AA)
            t6 = torch.stack([theta61, theta62, theta61, theta62, theta61, theta62, theta61, theta62], 0)
            # t6.register_hook(save_grad('t6'))
            # print("[grads]t6:", grads)
            # print(t6, -BB + torch.sqrt(BB**2 - 4*AA*CC), 2*AA)
        elif BB**2 - 4*AA*CC == 0:
            theta61 = atan2(-BB, 2*AA)
            t6 = torch.stack([theta61, theta61, theta61, theta61, theta61, theta61, theta61, theta61], 0)
            # t6.register_hook(save_grad('t6'))
            # print("[grads]t6:", grads)
            # print("角61推出来了", t6)
        else:
            # angle_solution = (abs(4*AA*CC - BB**2) - torch.tensor([0])) * 1e+14 * 100
            angle_solution = torch.unsqueeze((4*AA*CC - BB**2), 0) * 1e+14 * 100
            num_Error1 += 1
            # print("角62推出来了", angle_solution, AA, BB, CC, BB**2 - 4*AA*CC)

            return angle_solution, num_Error1, num_Error2, the_NANLOSS_of_illegal_solution_with_num_and_Nan


    # 求角4
    DD1 = -(oy*cos(theta11)*cos(t6[0]) + ny*cos(theta11)*sin(t6[0]) - ox*sin(theta11)*cos(t6[0]) - nx*sin(theta11)*sin(t6[0]))
    DD2 = -(oy*cos(theta11)*cos(t6[1]) + ny*cos(theta11)*sin(t6[1]) - ox*sin(theta11)*cos(t6[1]) - nx*sin(theta11)*sin(t6[1]))
    # print(DD1, DD2)
    # if DD1 > 1:
    #     print(DD1)
    # if DD2 > 1:
    #     print(DD2)
    theta41 = torch.acos(DD1)
    theta42 = torch.acos(DD2)
    t4 = torch.stack([theta41, theta41, theta42, theta42, theta41, theta41, theta42, theta42], 0)

    # 求角5
    EPSILON = 1e-7
    for ii in range(3):
        if sin(t4[ii]) < EPSILON:
            num_Error1 += 1
            angle_solution = t4[ii] * 100 - torch.tensor([0])
            # print("角5{}推出来了".format(ii), angle_solution)

            return angle_solution, num_Error1, num_Error2, the_NANLOSS_of_illegal_solution_with_num_and_Nan

    EE1 = (ax*sin(theta11) - ay*cos(theta11)) / sin(t4[0])
    EE2 = (ax*sin(theta11) - ay*cos(theta11)) / sin(t4[2])
    # print(EE1, EE2)
    # if EE1 > 1:
    #     print(EE1)
    # if EE2 > 1:
    #     print(EE2)
    theta51 = torch.asin(EE1)
    theta52 = torch.asin(EE2)
    t5 = torch.stack([theta51, theta52, theta51, theta52, theta51, theta52, theta51, theta52], 0)
    
    # 求角2
    for iii in range(2):
        if cos(t5[iii]) < EPSILON:
            num_Error1 += 1
            angle_solution = t5[iii] * 100 - torch.tensor([0])
            # print("角2{}推出来了".format(iii), angle_solution)

            return angle_solution, num_Error1, num_Error2, the_NANLOSS_of_illegal_solution_with_num_and_Nan

    FF1 = (oz*cos(t6[0]) + nz*sin(t6[0])) / sin(t4[0])
    FF2 = (oz*cos(t6[0]) + nz*sin(t6[0])) / sin(t4[2])
    FF3 = (oz*cos(t6[1]) + nz*sin(t6[1])) / sin(t4[0])
    FF4 = (oz*cos(t6[1]) + nz*sin(t6[1])) / sin(t4[2])

    GG1 = (az + FF1*cos(t4[0])*sin(t5[0])) / cos(t5[0])
    GG2 = (az + FF1*cos(t4[0])*sin(t5[1])) / cos(t5[1])

    GG3 = (az + FF2*cos(t4[2])*sin(t5[0])) / cos(t5[0])
    GG4 = (az + FF2*cos(t4[2])*sin(t5[1])) / cos(t5[1])

    GG5 = (az + FF3*cos(t4[0])*sin(t5[0])) / cos(t5[0])
    GG6 = (az + FF3*cos(t4[0])*sin(t5[1])) / cos(t5[1])

    GG7 = (az + FF4*cos(t4[2])*sin(t5[0])) / cos(t5[0])
    GG8 = (az + FF4*cos(t4[2])*sin(t5[1])) / cos(t5[1])
    # print(GG1, GG2, GG3, GG4, GG5, GG6, GG7, GG8)

    theta21 = torch.acos((pz - d[0] - az*d[5] - d[3]*GG1) / a[2])
    theta22 = torch.acos((pz - d[0] - az*d[5] - d[3]*GG2) / a[2])
    theta23 = torch.acos((pz - d[0] - az*d[5] - d[3]*GG3) / a[2])
    theta24 = torch.acos((pz - d[0] - az*d[5] - d[3]*GG4) / a[2])
    theta25 = torch.acos((pz - d[0] - az*d[5] - d[3]*GG5) / a[2])
    theta26 = torch.acos((pz - d[0] - az*d[5] - d[3]*GG6) / a[2])
    theta27 = torch.acos((pz - d[0] - az*d[5] - d[3]*GG7) / a[2])
    theta28 = torch.acos((pz - d[0] - az*d[5] - d[3]*GG8) / a[2])

    t2 = torch.stack([theta21, theta22, theta23, theta24, theta25, theta26, theta27, theta28], 0)

    save_what_caused_Error2_as_Nan.append((pz - d[0] - az*d[5] - d[3]*GG1) / a[2])
    save_what_caused_Error2_as_Nan.append((pz - d[0] - az*d[5] - d[3]*GG2) / a[2])
    save_what_caused_Error2_as_Nan.append((pz - d[0] - az*d[5] - d[3]*GG3) / a[2])
    save_what_caused_Error2_as_Nan.append((pz - d[0] - az*d[5] - d[3]*GG4) / a[2])
    save_what_caused_Error2_as_Nan.append((pz - d[0] - az*d[5] - d[3]*GG5) / a[2])
    save_what_caused_Error2_as_Nan.append((pz - d[0] - az*d[5] - d[3]*GG6) / a[2])
    save_what_caused_Error2_as_Nan.append((pz - d[0] - az*d[5] - d[3]*GG7) / a[2])
    save_what_caused_Error2_as_Nan.append((pz - d[0] - az*d[5] - d[3]*GG8) / a[2])

    # t2.register_hook(save_grad('t2'))
    # print("[grads]t2:", grads["t2"])
    # print(save_what_caused_Error2_as_Nan)

    nan_index = torch.isnan(t2).nonzero()
    # print(nan_index)
    for i in nan_index:
        the_NANLOSS_of_illegal_solution_with_num_and_Nan = the_NANLOSS_of_illegal_solution_with_num_and_Nan + \
                                                           (abs(save_what_caused_Error2_as_Nan[i]) - torch.tensor([1])) * 200

    if len(nan_index) == 8:
        GG = (GG1 + GG2 + GG3 + GG4 + GG5 + GG6 + GG7 + GG8) / 8
        angle_solution = (abs((pz - d[0] - az*d[5] - d[3]*GG) / a[2]) - torch.tensor([1])) * 100

        num_Error2 += 1

        # print("从角2出去的angle_solution: ", angle_solution)

        return angle_solution, num_Error1, num_Error2, the_NANLOSS_of_illegal_solution_with_num_and_Nan

    else:
        pass

    # 求角3
    theta2_3_1 = atan2(FF1, GG1)
    theta2_3_2 = atan2(FF1, GG2)
    theta2_3_3 = atan2(FF2, GG3)
    theta2_3_4 = atan2(FF2, GG4)
    theta2_3_5 = atan2(FF3, GG5)
    theta2_3_6 = atan2(FF3, GG6)
    theta2_3_7 = atan2(FF4, GG7)
    theta2_3_8 = atan2(FF4, GG8)

    theta31 = theta2_3_1 - theta21
    theta32 = theta2_3_2 - theta22
    theta33 = theta2_3_3 - theta23
    theta34 = theta2_3_4 - theta24
    theta35 = theta2_3_5 - theta25
    theta36 = theta2_3_6 - theta26
    theta37 = theta2_3_7 - theta27
    theta38 = theta2_3_8 - theta28

    t3 = torch.stack([theta31, theta32, theta33, theta34, theta35, theta36, theta37, theta38], 0)

    angle_solution = torch.stack([t1, t2, t3, t4, t5, t6], 0)
    angle_solution = torch.t(angle_solution)
    # print("角3推出来了", angle_solution)

    return angle_solution, num_Error1, num_Error2, the_NANLOSS_of_illegal_solution_with_num_and_Nan


def calculate_IK_test(input_tar, MLP_output_base, a, d, alpha):

    IK_test_incorrect = 0
    save_what_caused_Error2_as_Nan = []
    the_NANLOSS_of_illegal_solution_with_num_and_Nan = torch.tensor([0.0], requires_grad=True)

    TT = torch.mm(transpose(MLP_output_base), input_tar)
    nx = TT[0, 0]
    ny = TT[1, 0]
    nz = TT[2, 0]
    ox = TT[0, 1]
    oy = TT[1, 1]
    oz = TT[2, 1]
    ax = TT[0, 2]
    ay = TT[1, 2]
    az = TT[2, 2]
    px = TT[0, 3]
    py = TT[1, 3]
    pz = TT[2, 3]
    # print(TT, MLP_output_base)

    # 求角1
    m = py - ay*d[5]
    n = px - ax*d[5]
    theta11 = atan2(m, n)
    t1 = torch.stack([theta11, theta11, theta11, theta11, theta11, theta11, theta11, theta11], 0)

    # 求角6
    # AA = (nx**2 + ny**2 + nz**2 - 1) * 1e+7
    # BB = (2*nx*ox + 2*ny*oy + 2*oz*nz) * 1e+7
    # CC = (ox**2 + oy**2 + oz**2 - 1) * 1e+7
    AA = nx**2 + ny**2 + nz**2 - 1
    BB = 2*nx*ox + 2*ny*oy + 2*oz*nz
    CC = ox**2 + oy**2 + oz**2 - 1
    if AA == 0:
        if BB == 0:
            IK_test_incorrect += 1
            angle_solution = torch.tensor([100.0], requires_grad=True)

            return angle_solution
        else:
            theta61 = atan2(-CC, BB)
            t6 = torch.stack([theta61, theta61, theta61, theta61, theta61, theta61, theta61, theta61], 0)
            # t6.register_hook(save_grad('t6'))
            # print("[grads]t6:", grads)
            # print("角61推出来了", t6)

    else:

        if BB**2 - 4*AA*CC > 0:
            theta61 = atan2(-BB + torch.sqrt(BB**2 - 4*AA*CC), 2*AA)
            theta62 = atan2(-BB - torch.sqrt(BB**2 - 4*AA*CC), 2*AA)
            t6 = torch.stack([theta61, theta62, theta61, theta62, theta61, theta62, theta61, theta62], 0)
            # t6.register_hook(save_grad('t6'))
            # print("[grads]t6:", grads)
            # print(t6, -BB + torch.sqrt(BB**2 - 4*AA*CC), 2*AA)
        elif BB**2 - 4*AA*CC == 0:
            theta61 = atan2(-BB, 2*AA)
            t6 = torch.stack([theta61, theta61, theta61, theta61, theta61, theta61, theta61, theta61], 0)
            # t6.register_hook(save_grad('t6'))
            # print("[grads]t6:", grads)
            # print("角61推出来了", t6)
        else:
            # angle_solution = (abs(4*AA*CC - BB**2) - torch.tensor([0])) * 1e+14 * 100
            angle_solution = torch.unsqueeze((4*AA*CC - BB**2), 0) * 1e+14 * 100
            IK_test_incorrect += 1
            # print("角62推出来了", angle_solution, AA, BB, CC, BB**2 - 4*AA*CC)

            return angle_solution


    # 求角4
    DD1 = -(oy*cos(theta11)*cos(t6[0]) + ny*cos(theta11)*sin(t6[0]) - ox*sin(theta11)*cos(t6[0]) - nx*sin(theta11)*sin(t6[0]))
    DD2 = -(oy*cos(theta11)*cos(t6[1]) + ny*cos(theta11)*sin(t6[1]) - ox*sin(theta11)*cos(t6[1]) - nx*sin(theta11)*sin(t6[1]))
    # print(DD1, DD2)
    # if DD1 > 1:
    #     print(DD1)
    # if DD2 > 1:
    #     print(DD2)
    theta41 = torch.acos(DD1)
    theta42 = torch.acos(DD2)
    t4 = torch.stack([theta41, theta41, theta42, theta42, theta41, theta41, theta42, theta42], 0)

    # 求角5
    EPSILON = 1e-7
    for ii in range(3):
        if sin(t4[ii]) < EPSILON:
            IK_test_incorrect += 1
            angle_solution = t4[ii] * 100 - torch.tensor([0])
            # print("角5{}推出来了".format(ii), angle_solution)

            return angle_solution

    EE1 = (ax*sin(theta11) - ay*cos(theta11)) / sin(t4[0])
    EE2 = (ax*sin(theta11) - ay*cos(theta11)) / sin(t4[2])
    # print(EE1, EE2)
    # if EE1 > 1:
    #     print(EE1)
    # if EE2 > 1:
    #     print(EE2)
    theta51 = torch.asin(EE1)
    theta52 = torch.asin(EE2)
    t5 = torch.stack([theta51, theta52, theta51, theta52, theta51, theta52, theta51, theta52], 0)
    
    # 求角2
    for iii in range(2):
        if cos(t5[iii]) < EPSILON:
            IK_test_incorrect += 1
            angle_solution = t5[iii] * 100 - torch.tensor([0])
            # print("角2{}推出来了".format(iii), angle_solution)

            return angle_solution

    FF1 = (oz*cos(t6[0]) + nz*sin(t6[0])) / sin(t4[0])
    FF2 = (oz*cos(t6[0]) + nz*sin(t6[0])) / sin(t4[2])
    FF3 = (oz*cos(t6[1]) + nz*sin(t6[1])) / sin(t4[0])
    FF4 = (oz*cos(t6[1]) + nz*sin(t6[1])) / sin(t4[2])

    GG1 = (az + FF1*cos(t4[0])*sin(t5[0])) / cos(t5[0])
    GG2 = (az + FF1*cos(t4[0])*sin(t5[1])) / cos(t5[1])

    GG3 = (az + FF2*cos(t4[2])*sin(t5[0])) / cos(t5[0])
    GG4 = (az + FF2*cos(t4[2])*sin(t5[1])) / cos(t5[1])

    GG5 = (az + FF3*cos(t4[0])*sin(t5[0])) / cos(t5[0])
    GG6 = (az + FF3*cos(t4[0])*sin(t5[1])) / cos(t5[1])

    GG7 = (az + FF4*cos(t4[2])*sin(t5[0])) / cos(t5[0])
    GG8 = (az + FF4*cos(t4[2])*sin(t5[1])) / cos(t5[1])
    # print(GG1, GG2, GG3, GG4, GG5, GG6, GG7, GG8)

    theta21 = torch.acos((pz - d[0] - az*d[5] - d[3]*GG1) / a[2])
    theta22 = torch.acos((pz - d[0] - az*d[5] - d[3]*GG2) / a[2])
    theta23 = torch.acos((pz - d[0] - az*d[5] - d[3]*GG3) / a[2])
    theta24 = torch.acos((pz - d[0] - az*d[5] - d[3]*GG4) / a[2])
    theta25 = torch.acos((pz - d[0] - az*d[5] - d[3]*GG5) / a[2])
    theta26 = torch.acos((pz - d[0] - az*d[5] - d[3]*GG6) / a[2])
    theta27 = torch.acos((pz - d[0] - az*d[5] - d[3]*GG7) / a[2])
    theta28 = torch.acos((pz - d[0] - az*d[5] - d[3]*GG8) / a[2])

    t2 = torch.stack([theta21, theta22, theta23, theta24, theta25, theta26, theta27, theta28], 0)

    save_what_caused_Error2_as_Nan.append((pz - d[0] - az*d[5] - d[3]*GG1) / a[2])
    save_what_caused_Error2_as_Nan.append((pz - d[0] - az*d[5] - d[3]*GG2) / a[2])
    save_what_caused_Error2_as_Nan.append((pz - d[0] - az*d[5] - d[3]*GG3) / a[2])
    save_what_caused_Error2_as_Nan.append((pz - d[0] - az*d[5] - d[3]*GG4) / a[2])
    save_what_caused_Error2_as_Nan.append((pz - d[0] - az*d[5] - d[3]*GG5) / a[2])
    save_what_caused_Error2_as_Nan.append((pz - d[0] - az*d[5] - d[3]*GG6) / a[2])
    save_what_caused_Error2_as_Nan.append((pz - d[0] - az*d[5] - d[3]*GG7) / a[2])
    save_what_caused_Error2_as_Nan.append((pz - d[0] - az*d[5] - d[3]*GG8) / a[2])

    # t2.register_hook(save_grad('t2'))
    # print("[grads]t2:", grads["t2"])
    # print(save_what_caused_Error2_as_Nan)

    nan_index = torch.isnan(t2).nonzero()
    # print(nan_index)
    for i in nan_index:
        the_NANLOSS_of_illegal_solution_with_num_and_Nan = the_NANLOSS_of_illegal_solution_with_num_and_Nan + \
                                                           (abs(save_what_caused_Error2_as_Nan[i]) - torch.tensor([1])) * 200

    if len(nan_index) == 8:
        GG = (GG1 + GG2 + GG3 + GG4 + GG5 + GG6 + GG7 + GG8) / 8
        angle_solution = (abs((pz - d[0] - az*d[5] - d[3]*GG) / a[2]) - torch.tensor([1])) * 100

        IK_test_incorrect += 1

        # print("从角2出去的angle_solution: ", angle_solution)

        return angle_solution

    else:
        pass

    # 求角3
    theta2_3_1 = atan2(FF1, GG1)
    theta2_3_2 = atan2(FF1, GG2)
    theta2_3_3 = atan2(FF2, GG3)
    theta2_3_4 = atan2(FF2, GG4)
    theta2_3_5 = atan2(FF3, GG5)
    theta2_3_6 = atan2(FF3, GG6)
    theta2_3_7 = atan2(FF4, GG7)
    theta2_3_8 = atan2(FF4, GG8)

    theta31 = theta2_3_1 - theta21
    theta32 = theta2_3_2 - theta22
    theta33 = theta2_3_3 - theta23
    theta34 = theta2_3_4 - theta24
    theta35 = theta2_3_5 - theta25
    theta36 = theta2_3_6 - theta26
    theta37 = theta2_3_7 - theta27
    theta38 = theta2_3_8 - theta28

    t3 = torch.stack([theta31, theta32, theta33, theta34, theta35, theta36, theta37, theta38], 0)

    angle_solution = torch.stack([t1, t2, t3, t4, t5, t6], 0)
    angle_solution = torch.t(angle_solution)
    # print("角3推出来了", angle_solution)

    return angle_solution