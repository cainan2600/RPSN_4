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

    # 求角1
    m = py - ay*d[5]
    n = px - ax*d[5]
    theta11 = atan2(m, n)
    t1 = torch.stack([theta11, theta11, theta11, theta11, theta11, theta11, theta11, theta11], 0)

    # 求角6
    AA = nx**2 + ny**2 + nz**2 - 1
    BB = 2*nx*ox + 2*ny*oy + 2*oz*nz
    CC = ox**2 + oy**2 + oz**2 - 1
    if AA == 0:
        num_Error2 += 1

        # return angle_solution, num_Error1, num_Error2, the_NANLOSS_of_illegal_solution_with_num_and_Nan
    else:
        if BB**2 - 4*AA*CC >= 0:
            theta61 = atan2(-BB + torch.sqrt(BB**2 - 4*AA*CC), 2*AA)
            theta62 = atan2(-BB - torch.sqrt(BB**2 - 4*AA*CC), 2*AA)
            t6 = torch.stack([theta61, theta62, theta61, theta62, theta61, theta62, theta61, theta62], 0)
        else:
            angle_solution = torch.unsqueeze((BB**2 - 4*AA*CC), 0) * 100
            num_Error1 += 1

            return angle_solution, num_Error1, num_Error2, the_NANLOSS_of_illegal_solution_with_num_and_Nan
    
    # 求角4
    DD1 = -(oy*cos(theta11)*cos(t6[0]) + ny*cos(theta11)*sin(t6[0]) - ox*sin(theta11)*cos(t6[0]) - nx*sin(theta11)*sin(t6[0]))
    DD2 = -(oy*cos(theta11)*cos(t6[1]) + ny*cos(theta11)*sin(t6[1]) - ox*sin(theta11)*cos(t6[1]) - nx*sin(theta11)*sin(t6[1]))
    theta41 = torch.acos(DD1)
    theta42 = torch.acos(DD2)
    t4 = torch.stack([theta41, theta41, theta42, theta42, theta41, theta41, theta42, theta42], 0)

    # 求角5
    for i in 2:
        if sin(t4[i]) == 0:
            num_Error1 += 1
            return angle_solution, num_Error1, num_Error2, the_NANLOSS_of_illegal_solution_with_num_and_Nan
        else:
            EE1 = (ax*sin(theta11) - ay*cos(theta11)) / sin(t4[0])
            EE2 = (ax*sin(theta11) - ay*cos(theta11)) / sin(t4[2])
            theta51 = torch.asin(EE1)
            theta52 = torch.asin(EE2)
            t5 = torch.stack([theta51, theta52, theta51, theta52, theta51, theta52, theta51, theta52], 0)
    
    # 求角2
    FF1 = (oz*cos(t6[0]) + nz*sin(t6[0])) / sin(t4[0])
    FF2 = (oz*cos(t6[0]) + nz*sin(t6[0])) / sin(t4[2])
    FF3 = (oz*cos(t6[1]) + nz*sin(t6[1])) / sin(t4[0])
    FF4 = (oz*cos(t6[1]) + nz*sin(t6[1])) / sin(t4[2])

    GG1 = (az + FF1*cos(t4[0])*sin(t5[0])) / cos(t5[0])
    GG2 = (az + FF1*cos(t4[0])*sin(t5[1])) / cos(t5[2])

    GG3 = (az + FF2*cos(t4[2])*sin(t5[0])) / cos(t5[0])
    GG4 = (az + FF2*cos(t4[2])*sin(t5[1])) / cos(t5[2])

    GG5 = (az + FF3*cos(t4[0])*sin(t5[0])) / cos(t5[0])
    GG6 = (az + FF3*cos(t4[0])*sin(t5[1])) / cos(t5[2])

    GG7 = (az + FF4*cos(t4[2])*sin(t5[0])) / cos(t5[0])
    GG8 = (az + FF4*cos(t4[2])*sin(t5[1])) / cos(t5[2])

    theta21 = torch.acos((pz - d1 - az*d6 - d4*GG1) / a3)
    theta22 = torch.acos((pz - d1 - az*d6 - d4*GG2) / a3)
    theta23 = torch.acos((pz - d1 - az*d6 - d4*GG3) / a3)
    theta24 = torch.acos((pz - d1 - az*d6 - d4*GG4) / a3)
    theta25 = torch.acos((pz - d1 - az*d6 - d4*GG5) / a3)
    theta26 = torch.acos((pz - d1 - az*d6 - d4*GG6) / a3)
    theta27 = torch.acos((pz - d1 - az*d6 - d4*GG7) / a3)
    theta28 = torch.acos((pz - d1 - az*d6 - d4*GG8) / a3)

    t2 = torch.stack([theta21, theta22, theta23, theta24, theta25, theta26, theta27, theta28], 0)

    # 求角3
    theta2_3_1 = atan2(FF1 / GG1)
    theta2_3_2 = atan2(FF1 / GG2)
    theta2_3_3 = atan2(FF2 / GG3)
    theta2_3_4 = atan2(FF2 / GG4)
    theta2_3_5 = atan2(FF3 / GG5)
    theta2_3_6 = atan2(FF3 / GG6)
    theta2_3_7 = atan2(FF4 / GG7)
    theta2_3_8 = atan2(FF4 / GG8)

    theta31 = theta2_3_1 - theta21
    theta32 = theta2_3_1 - theta22
    theta33 = theta2_3_1 - theta23
    theta34 = theta2_3_1 - theta24
    theta35 = theta2_3_1 - theta25
    theta36 = theta2_3_1 - theta26
    theta37 = theta2_3_1 - theta27
    theta38 = theta2_3_1 - theta28

    t3 = torch.stack([theta31, theta32, theta33, theta34, theta35, theta36, theta37, theta38], 0)

    return angle_solution, num_Error1, num_Error2, the_NANLOSS_of_illegal_solution_with_num_and_Nan


def calculate_IK_test(input_tar, MLP_output_base, a, d, alpha):

    IK_test_incorrect = 0

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

    # 求角1
    m = py - ay*d[5]
    n = px - ax*d[5]
    theta11 = atan2(m, n)
    t1 = torch.stack([theta11, theta11, theta11, theta11, theta11, theta11, theta11, theta11], 0)

    # 求角6
    AA = nx**2 + ny**2 + nz**2 - 1
    BB = 2*nx*ox + 2*ny*oy + 2*oz*nz
    CC = ox**2 + oy**2 + oz**2 - 1
    if AA == 0:
        IK_test_incorrect += 1

        # return angle_solution, num_Error1, num_Error2, the_NANLOSS_of_illegal_solution_with_num_and_Nan
    else:
        if BB**2 - 4*AA*CC >= 0:
            theta61 = atan2(-BB + torch.sqrt(BB**2 - 4*AA*CC), 2*AA)
            theta62 = atan2(-BB - torch.sqrt(BB**2 - 4*AA*CC), 2*AA)
            t6 = torch.stack([theta61, theta62, theta61, theta62, theta61, theta62, theta61, theta62], 0)
        else:
            angle_solution = torch.unsqueeze((BB**2 - 4*AA*CC), 0) * 100
            IK_test_incorrect += 1

            return angle_solution
    
    # 求角4
    DD1 = -(oy*cos(theta11)*cos(t6[0]) + ny*cos(theta11)*sin(t6[0]) - ox*sin(theta11)*cos(t6[0]) - nx*sin(theta11)*sin(t6[0]))
    DD2 = -(oy*cos(theta11)*cos(t6[1]) + ny*cos(theta11)*sin(t6[1]) - ox*sin(theta11)*cos(t6[1]) - nx*sin(theta11)*sin(t6[1]))
    theta41 = torch.acos(DD1)
    theta42 = torch.acos(DD2)
    t4 = torch.stack([theta41, theta41, theta42, theta42, theta41, theta41, theta42, theta42], 0)

    # 求角5
    for i in 2:
        if sin(t4[i]) == 0:
            IK_test_incorrect += 1

            return angle_solution
        else:
            EE1 = (ax*sin(theta11) - ay*cos(theta11)) / sin(t4[0])
            EE2 = (ax*sin(theta11) - ay*cos(theta11)) / sin(t4[2])
            theta51 = torch.asin(EE1)
            theta52 = torch.asin(EE2)
            t5 = torch.stack([theta51, theta52, theta51, theta52, theta51, theta52, theta51, theta52], 0)
    
    # 求角2
    FF1 = (oz*cos(t6[0]) + nz*sin(t6[0])) / sin(t4[0])
    FF2 = (oz*cos(t6[0]) + nz*sin(t6[0])) / sin(t4[2])
    FF3 = (oz*cos(t6[1]) + nz*sin(t6[1])) / sin(t4[0])
    FF4 = (oz*cos(t6[1]) + nz*sin(t6[1])) / sin(t4[2])

    GG1 = (az + FF1*cos(t4[0])*sin(t5[0])) / cos(t5[0])
    GG2 = (az + FF1*cos(t4[0])*sin(t5[1])) / cos(t5[2])

    GG3 = (az + FF2*cos(t4[2])*sin(t5[0])) / cos(t5[0])
    GG4 = (az + FF2*cos(t4[2])*sin(t5[1])) / cos(t5[2])

    GG5 = (az + FF3*cos(t4[0])*sin(t5[0])) / cos(t5[0])
    GG6 = (az + FF3*cos(t4[0])*sin(t5[1])) / cos(t5[2])

    GG7 = (az + FF4*cos(t4[2])*sin(t5[0])) / cos(t5[0])
    GG8 = (az + FF4*cos(t4[2])*sin(t5[1])) / cos(t5[2])

    theta21 = torch.acos((pz - d1 - az*d6 - d4*GG1) / a3)
    theta22 = torch.acos((pz - d1 - az*d6 - d4*GG2) / a3)
    theta23 = torch.acos((pz - d1 - az*d6 - d4*GG3) / a3)
    theta24 = torch.acos((pz - d1 - az*d6 - d4*GG4) / a3)
    theta25 = torch.acos((pz - d1 - az*d6 - d4*GG5) / a3)
    theta26 = torch.acos((pz - d1 - az*d6 - d4*GG6) / a3)
    theta27 = torch.acos((pz - d1 - az*d6 - d4*GG7) / a3)
    theta28 = torch.acos((pz - d1 - az*d6 - d4*GG8) / a3)

    t2 = torch.stack([theta21, theta22, theta23, theta24, theta25, theta26, theta27, theta28], 0)

    # 求角3
    theta2_3_1 = atan2(FF1 / GG1)
    theta2_3_2 = atan2(FF1 / GG2)
    theta2_3_3 = atan2(FF2 / GG3)
    theta2_3_4 = atan2(FF2 / GG4)
    theta2_3_5 = atan2(FF3 / GG5)
    theta2_3_6 = atan2(FF3 / GG6)
    theta2_3_7 = atan2(FF4 / GG7)
    theta2_3_8 = atan2(FF4 / GG8)

    theta31 = theta2_3_1 - theta21
    theta32 = theta2_3_1 - theta22
    theta33 = theta2_3_1 - theta23
    theta34 = theta2_3_1 - theta24
    theta35 = theta2_3_1 - theta25
    theta36 = theta2_3_1 - theta26
    theta37 = theta2_3_1 - theta27
    theta38 = theta2_3_1 - theta28

    t3 = torch.stack([theta31, theta32, theta33, theta34, theta35, theta36, theta37, theta38], 0)

    return angle_solution