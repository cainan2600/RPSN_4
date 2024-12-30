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
    m = d[5] * ay - py
    n = ax * d[5] - px
    if m ** 2 + n ** 2 - (d[3]) ** 2 >= 0:
        theta11 = atan2(m, n) - atan2(d[3], torch.sqrt((m ** 2 + n ** 2 - (d[3]) ** 2)))
        theta12 = atan2(m, n) - atan2(d[3], -torch.sqrt((m ** 2 + n ** 2 - (d[3]) ** 2)))

        t1 = torch.cat([theta11.repeat(4), theta12.repeat(4)], dim=0)

    else:
        angle_solution = torch.unsqueeze(((d[3]) ** 2 - m ** 2 - n ** 2), 0) * 100

        num_Error1 += 1

        return angle_solution, num_Error1, num_Error2, the_NANLOSS_of_illegal_solution_with_num_and_Nan

    # 求角5
    theta51 = torch.acos(ax * sin(theta11) - ay * cos(theta11))
    theta52 = -torch.acos(ax * sin(theta11) - ay * cos(theta11))
    theta53 = torch.acos(ax * sin(theta12) - ay * cos(theta12))
    theta54 = -torch.acos(ax * sin(theta12) - ay * cos(theta12))
    t5 = torch.stack([theta51, theta51, theta52, theta52, theta53, theta53, theta54, theta54], 0)
    # print(ax * sin(theta11) - ay * cos(theta11))

    # 求角6
    mm = nx * sin(t1[0]) - ny * cos(t1[0])
    nn = ox * sin(t1[0]) - oy * cos(t1[0])
    t61 = atan2(mm, nn) - atan2(sin(t5[0]), torch.tensor(0.0))

    mm = nx * sin(t1[1]) - ny * cos(t1[1])
    nn = ox * sin(t1[1]) - oy * cos(t1[1])
    t62 = atan2(mm, nn) - atan2(sin(t5[1]), torch.tensor(0.0))

    mm = nx * sin(t1[2]) - ny * cos(t1[2])
    nn = ox * sin(t1[2]) - oy * cos(t1[2])
    t63 = atan2(mm, nn) - atan2(sin(t5[2]), torch.tensor(0.0))

    mm = nx * sin(t1[3]) - ny * cos(t1[3])
    nn = ox * sin(t1[3]) - oy * cos(t1[3])
    t64 = atan2(mm, nn) - atan2(sin(t5[3]), torch.tensor(0.0))

    mm = nx * sin(t1[4]) - ny * cos(t1[4])
    nn = ox * sin(t1[4]) - oy * cos(t1[4])
    t65 = atan2(mm, nn) - atan2(sin(t5[4]), torch.tensor(0.0))

    mm = nx * sin(t1[5]) - ny * cos(t1[5])
    nn = ox * sin(t1[5]) - oy * cos(t1[5])
    t66 = atan2(mm, nn) - atan2(sin(t5[5]), torch.tensor(0.0))

    mm = nx * sin(t1[6]) - ny * cos(t1[6])
    nn = ox * sin(t1[6]) - oy * cos(t1[6])
    t67 = atan2(mm, nn) - atan2(sin(t5[6]), torch.tensor(0.0))

    mm = nx * sin(t1[7]) - ny * cos(t1[7])
    nn = ox * sin(t1[7]) - oy * cos(t1[7])
    t68 = atan2(mm, nn) - atan2(sin(t5[7]), torch.tensor(0.0))
    t6 = torch.stack([t61, t62, t63, t64, t65, t66, t67, t68], 0)
    # print(t6)

    # 求角3

    m = [0, 0, 0, 0, 0, 0, 0, 0]
    n = [0, 0, 0, 0, 0, 0, 0, 0]
    for i in range(8):
        m[i] = d[4] * (sin(t6[i]) * (nx * cos(t1[i]) + ny * sin(t1[i])) + cos(t6[i]) * (
                ox * cos(t1[i]) + oy * sin(t1[i]))) - d[5] * (ax * cos(t1[i]) + ay * sin(t1[i])) + px * cos(
            t1[i]) + py * sin(t1[i])
        n[i] = pz - d[0] - az * d[5] + d[4] * (oz * cos(t6[i]) + nz * sin(t6[i]))

    # try:
    t31 = torch.acos((m[0] ** 2 + n[0] ** 2 - a[1] ** 2 - a[2] ** 2) / (2 * a[1] * a[2]))
    t32 = -torch.acos((m[0] ** 2 + n[0] ** 2 - a[1] ** 2 - a[2] ** 2) / (2 * a[1] * a[2]))
    t33 = torch.acos((m[2] ** 2 + n[2] ** 2 - a[1] ** 2 - a[2] ** 2) / (2 * a[1] * a[2]))
    t34 = -torch.acos((m[2] ** 2 + n[2] ** 2 - a[1] ** 2 - a[2] ** 2) / (2 * a[1] * a[2]))
    t35 = torch.acos((m[4] ** 2 + n[4] ** 2 - a[1] ** 2 - a[2] ** 2) / (2 * a[1] * a[2]))
    t36 = -torch.acos((m[4] ** 2 + n[4] ** 2 - a[1] ** 2 - a[2] ** 2) / (2 * a[1] * a[2]))
    t37 = torch.acos((m[6] ** 2 + n[6] ** 2 - a[1] ** 2 - a[2] ** 2) / (2 * a[1] * a[2]))
    t38 = -torch.acos((m[6] ** 2 + n[6] ** 2 - a[1] ** 2 - a[2] ** 2) / (2 * a[1] * a[2]))
    # print(m[0] ,n[0], a[1], a[2])
    # t3 = torch.stack([t31-math.pi/2, t32-math.pi/2, t33-math.pi/2, t34-math.pi/2, t35-math.pi/2, t36-math.pi/2, t37-math.pi/2, t38-math.pi/2], 0)
    t3 = torch.stack([t31, t32, t33, t34, t35, t36, t37, t38], 0)

    save_what_caused_Error2_as_Nan.append((m[0] ** 2 + n[0] ** 2 - a[1] ** 2 - a[2] ** 2) / (2 * a[1] * a[2]))
    save_what_caused_Error2_as_Nan.append((m[0] ** 2 + n[0] ** 2 - a[1] ** 2 - a[2] ** 2) / (2 * a[1] * a[2]))
    save_what_caused_Error2_as_Nan.append((m[2] ** 2 + n[2] ** 2 - a[1] ** 2 - a[2] ** 2) / (2 * a[1] * a[2]))
    save_what_caused_Error2_as_Nan.append((m[2] ** 2 + n[2] ** 2 - a[1] ** 2 - a[2] ** 2) / (2 * a[1] * a[2]))
    save_what_caused_Error2_as_Nan.append((m[4] ** 2 + n[4] ** 2 - a[1] ** 2 - a[2] ** 2) / (2 * a[1] * a[2]))
    save_what_caused_Error2_as_Nan.append((m[4] ** 2 + n[4] ** 2 - a[1] ** 2 - a[2] ** 2) / (2 * a[1] * a[2]))
    save_what_caused_Error2_as_Nan.append((m[6] ** 2 + n[6] ** 2 - a[1] ** 2 - a[2] ** 2) / (2 * a[1] * a[2]))
    save_what_caused_Error2_as_Nan.append((m[6] ** 2 + n[6] ** 2 - a[1] ** 2 - a[2] ** 2) / (2 * a[1] * a[2]))

    nan_index = torch.isnan(t3).nonzero()
    for i in nan_index:
        the_NANLOSS_of_illegal_solution_with_num_and_Nan = the_NANLOSS_of_illegal_solution_with_num_and_Nan + \
                                                           (abs(save_what_caused_Error2_as_Nan[i]) - torch.tensor([1])) * 200

    if len(nan_index) == 8:
        aaabbb = nan_index[0].item()
        cccddd = (m[aaabbb] ** 2 + n[aaabbb] ** 2 - a[1] ** 2 - a[2] ** 2) / (2 * a[1] * a[2])
        angle_solution = (abs(cccddd) - torch.tensor([1])) * 100

        num_Error2 += 1

        # print("从角3出去的angle_solution: ", angle_solution)

        return angle_solution, num_Error1, num_Error2, the_NANLOSS_of_illegal_solution_with_num_and_Nan

    else:
        pass

    # 求角2
    t2 = [0, 0, 0, 0, 0, 0, 0, 0]
    s2 = [0, 0, 0, 0, 0, 0, 0, 0]
    c2 = [0, 0, 0, 0, 0, 0, 0, 0]
    for i in range(8):
        s2[i] = ((a[2] * cos(t3[i]) + a[1]) * n[i] - a[2] * sin(t3[i]) * m[i]) / (
                a[1] ** 2 + a[2] ** 2 + 2 * a[1] * a[2] * cos(t3[i]))
        c2[i] = (m[i] + (a[2] * sin(t3[i]) * s2[i])) / (a[2] * cos(t3[i]) + a[1])

    t20 = atan2(s2[0], c2[0])
    t21 = atan2(s2[1], c2[1])
    t22 = atan2(s2[2], c2[2])
    t23 = atan2(s2[3], c2[3])
    t24 = atan2(s2[4], c2[4])
    t25 = atan2(s2[5], c2[5])
    t26 = atan2(s2[6], c2[6])
    t27 = atan2(s2[7], c2[7])

    t2 = torch.stack([t20, t21, t22, t23, t24, t25, t26, t27], 0)

    # 求角4
    t4 = [0, 0, 0, 0, 0, 0, 0, 0]

    for i in range(8):
        t4[i] = atan2(
            -sin(t6[i]) * (nx * cos(t1[i]) + ny * sin(t1[i])) - cos(t6[i]) * (ox * cos(t1[i]) + oy * sin(t1[i])),
            oz * cos(t6[i]) + nz * sin(t6[i])) - t2[i] - t3[i]
    t4 = torch.stack([t4[0], t4[1], t4[2], t4[3], t4[4], t4[5], t4[6], t4[7]], 0)
    angle_solution = torch.stack([t1, t2, t3, t4, t5, t6], 0)
    angle_solution = torch.t(angle_solution)

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
    m = d[5] * ay - py
    n = ax * d[5] - px
    if m ** 2 + n ** 2 - (d[3]) ** 2 >= 0:
        theta11 = atan2(m, n) - atan2(d[3], torch.sqrt((m ** 2 + n ** 2 - (d[3]) ** 2)))
        theta12 = atan2(m, n) - atan2(d[3], -torch.sqrt((m ** 2 + n ** 2 - (d[3]) ** 2)))

        t1 = torch.cat([theta11.repeat(4), theta12.repeat(4)], dim=0)

    else:
        angle_solution = torch.unsqueeze(((d[3]) ** 2 - m ** 2 - n ** 2), 0)

        IK_test_incorrect += 1

        return angle_solution

    # 求角5
    theta51 = torch.acos(ax * sin(theta11) - ay * cos(theta11))
    theta52 = -torch.acos(ax * sin(theta11) - ay * cos(theta11))
    theta53 = torch.acos(ax * sin(theta12) - ay * cos(theta12))
    theta54 = -torch.acos(ax * sin(theta12) - ay * cos(theta12))
    t5 = torch.stack([theta51, theta51, theta52, theta52, theta53, theta53, theta54, theta54], 0)

    # 求角6
    mm = nx * sin(t1[0]) - ny * cos(t1[0])
    nn = ox * sin(t1[0]) - oy * cos(t1[0])
    t61 = atan2(mm, nn) - atan2(sin(t5[0]), torch.tensor(0.0))

    mm = nx * sin(t1[1]) - ny * cos(t1[1])
    nn = ox * sin(t1[1]) - oy * cos(t1[1])
    t62 = atan2(mm, nn) - atan2(sin(t5[1]), torch.tensor(0.0))

    mm = nx * sin(t1[2]) - ny * cos(t1[2])
    nn = ox * sin(t1[2]) - oy * cos(t1[2])
    t63 = atan2(mm, nn) - atan2(sin(t5[2]), torch.tensor(0.0))

    mm = nx * sin(t1[3]) - ny * cos(t1[3])
    nn = ox * sin(t1[3]) - oy * cos(t1[3])
    t64 = atan2(mm, nn) - atan2(sin(t5[3]), torch.tensor(0.0))

    mm = nx * sin(t1[4]) - ny * cos(t1[4])
    nn = ox * sin(t1[4]) - oy * cos(t1[4])
    t65 = atan2(mm, nn) - atan2(sin(t5[4]), torch.tensor(0.0))

    mm = nx * sin(t1[5]) - ny * cos(t1[5])
    nn = ox * sin(t1[5]) - oy * cos(t1[5])
    t66 = atan2(mm, nn) - atan2(sin(t5[5]), torch.tensor(0.0))

    mm = nx * sin(t1[6]) - ny * cos(t1[6])
    nn = ox * sin(t1[6]) - oy * cos(t1[6])
    t67 = atan2(mm, nn) - atan2(sin(t5[6]), torch.tensor(0.0))

    mm = nx * sin(t1[7]) - ny * cos(t1[7])
    nn = ox * sin(t1[7]) - oy * cos(t1[7])
    t68 = atan2(mm, nn) - atan2(sin(t5[7]), torch.tensor(0.0))
    t6 = torch.stack([t61, t62, t63, t64, t65, t66, t67, t68], 0)

    # 求角3

    m = [0, 0, 0, 0, 0, 0, 0, 0]
    n = [0, 0, 0, 0, 0, 0, 0, 0]
    for i in range(8):
        m[i] = d[4] * (sin(t6[i]) * (nx * cos(t1[i]) + ny * sin(t1[i])) + cos(t6[i]) * (
                ox * cos(t1[i]) + oy * sin(t1[i]))) - d[5] * (ax * cos(t1[i]) + ay * sin(t1[i])) + px * cos(
            t1[i]) + py * sin(t1[i])
        n[i] = pz - d[0] - az * d[5] + d[4] * (oz * cos(t6[i]) + nz * sin(t6[i]))


    # try:
    t31 = torch.acos((m[0] ** 2 + n[0] ** 2 - a[1] ** 2 - a[2] ** 2) / (2 * a[1] * a[2]))
    t32 = -torch.acos((m[0] ** 2 + n[0] ** 2 - a[1] ** 2 - a[2] ** 2) / (2 * a[1] * a[2]))
    t33 = torch.acos((m[2] ** 2 + n[2] ** 2 - a[1] ** 2 - a[2] ** 2) / (2 * a[1] * a[2]))
    t34 = -torch.acos((m[2] ** 2 + n[2] ** 2 - a[1] ** 2 - a[2] ** 2) / (2 * a[1] * a[2]))
    t35 = torch.acos((m[4] ** 2 + n[4] ** 2 - a[1] ** 2 - a[2] ** 2) / (2 * a[1] * a[2]))
    t36 = -torch.acos((m[4] ** 2 + n[4] ** 2 - a[1] ** 2 - a[2] ** 2) / (2 * a[1] * a[2]))
    t37 = torch.acos((m[6] ** 2 + n[6] ** 2 - a[1] ** 2 - a[2] ** 2) / (2 * a[1] * a[2]))
    t38 = -torch.acos((m[6] ** 2 + n[6] ** 2 - a[1] ** 2 - a[2] ** 2) / (2 * a[1] * a[2]))
    # t3 = torch.stack([t31-math.pi/2, t32-math.pi/2, t33-math.pi/2, t34-math.pi/2, t35-math.pi/2, t36-math.pi/2, t37-math.pi/2, t38-math.pi/2], 0)
    t3 = torch.stack([t31, t32, t33, t34, t35, t36, t37, t38], 0)

    nan_index = torch.isnan(t3).nonzero()


    if len(nan_index) == 8:
        aaabbb = nan_index[0].item()
        cccddd = (m[aaabbb] ** 2 + n[aaabbb] ** 2 - a[1] ** 2 - a[2] ** 2) / (2 * a[1] * a[2])

        angle_solution = (abs(cccddd) - torch.tensor([1])) * 100

        IK_test_incorrect += 1

        return angle_solution

    else:
        pass

    # 求角2
    t2 = [0, 0, 0, 0, 0, 0, 0, 0]
    s2 = [0, 0, 0, 0, 0, 0, 0, 0]
    c2 = [0, 0, 0, 0, 0, 0, 0, 0]
    for i in range(8):
        s2[i] = ((a[2] * cos(t3[i]) + a[1]) * n[i] - a[2] * sin(t3[i]) * m[i]) / (
                a[1] ** 2 + a[2] ** 2 + 2 * a[1] * a[2] * cos(t3[i]))
        c2[i] = (m[i] + (a[2] * sin(t3[i]) * s2[i])) / (a[2] * cos(t3[i]) + a[1])

    t20 = atan2(s2[0], c2[0])
    t21 = atan2(s2[1], c2[1])
    t22 = atan2(s2[2], c2[2])
    t23 = atan2(s2[3], c2[3])
    t24 = atan2(s2[4], c2[4])
    t25 = atan2(s2[5], c2[5])
    t26 = atan2(s2[6], c2[6])
    t27 = atan2(s2[7], c2[7])

    # t2 = torch.stack([t20+math.pi/2, t21+math.pi/2, t22+math.pi/2, t23+math.pi/2, t24+math.pi/2, t25+math.pi/2, t26+math.pi/2, t27+math.pi/2], 0)
    t2 = torch.stack([t20, t21, t22, t23, t24, t25, t26, t27], 0)

    # 求角4
    t4 = [0, 0, 0, 0, 0, 0, 0, 0]

    for i in range(8):
        t4[i] = atan2(
            -sin(t6[i]) * (nx * cos(t1[i]) + ny * sin(t1[i])) - cos(t6[i]) * (ox * cos(t1[i]) + oy * sin(t1[i])),
            oz * cos(t6[i]) + nz * sin(t6[i])) - t2[i] - t3[i]
    t4 = torch.stack([t4[0], t4[1], t4[2], t4[3], t4[4], t4[5], t4[6], t4[7]], 0)
    angle_solution = torch.stack([t1, t2, t3, t4, t5, t6], 0)
    angle_solution = torch.t(angle_solution)


    return angle_solution