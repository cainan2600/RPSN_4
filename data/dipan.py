import numpy as np
import matplotlib.pyplot as plt


# def generrate_yuanxin(x):
#     """
#     UR10e
#     """

#     # 桌子的尺寸（长和宽）
#     table_length = 4
#     table_width = 2.6

#     # 外延范围（距离桌子边缘不超过的距离）
#     outer_range = 0.4

#     num_left_right = int(0.18 * x)
#     num_top_bottom = int(0.32 * x)

#     # 用于存储所有符合要求的点的列表
#     points = []

#     # 在桌子左边外延区域生成点（x坐标小于0，y坐标在对应的范围内）
#     left_outer_x = np.random.uniform(-0.4, 0, size=num_left_right)
#     left_outer_y = np.random.uniform(0, 2.6, size=num_left_right)
#     left_outer_points = np.column_stack((left_outer_x, left_outer_y))
#     points.extend(left_outer_points)

#     # 在桌子右边外延区域生成点（x坐标大于桌子长度，y坐标在对应的范围内）
#     right_outer_x = np.random.uniform(4, 4.4, size=num_left_right)
#     right_outer_y = np.random.uniform(0, 2.6, size=num_left_right)
#     right_outer_points = np.column_stack((right_outer_x, right_outer_y))
#     points.extend(right_outer_points)

#     # 在桌子上边外延区域生成点（y坐标大于桌子宽度，x坐标在对应的范围内）
#     top_outer_x = np.random.uniform(-0.4, 4.4, size=num_top_bottom)
#     top_outer_y = np.random.uniform(-0.4, 0, size=num_top_bottom)
#     top_outer_points = np.column_stack((top_outer_x, top_outer_y))
#     points.extend(top_outer_points)

#     # 在桌子下边外延区域生成点（y坐标小于0，x坐标在对应的范围内）
#     bottom_outer_x = np.random.uniform(-0.4, 4.4, size=num_top_bottom)
#     bottom_outer_y = np.random.uniform(2.6, 3, size=num_top_bottom)
#     bottom_outer_points = np.column_stack((bottom_outer_x, bottom_outer_y))
#     points.extend(bottom_outer_points)

#     # 转换为numpy数组以便后续操作（可选，如果不需要对整体点集做统一的数组操作，可省略这步）
#     points = np.array(points)
#     np.random.shuffle(points)

#     return points
#     # print(points[0][0])

#     # # 以下是可视化部分，用于展示生成的点和桌子的位置关系，可按需注释掉
#     # plt.plot([0, table_length, table_length, 0, 0], [0, 0, table_width, table_width, 0], 'r')  # 绘制桌子边框
#     # plt.scatter(points[:, 0], points[:, 1])  # 绘制生成的点
#     # plt.xlim(-outer_range, table_length + outer_range)
#     # plt.ylim(-outer_range, table_width + outer_range)
#     # plt.show()

def generrate_yuanxin(x):
    """
    Realman
    """

    # 桌子的尺寸（长和宽）
    table_length = 2
    table_width = 0.8

    # 外延范围（距离桌子边缘不超过的距离）
    outer_range = 0.4

    num_left_right = int(0.2 * x)
    num_top_bottom = int(0.3 * x)

    # 用于存储所有符合要求的点的列表
    points = []

    # 在桌子左边外延区域生成点（x坐标小于0，y坐标在对应的范围内）
    left_outer_x = np.random.uniform(-1.4, -1.3, size=num_left_right)
    left_outer_y = np.random.uniform(-0.825, 0.825, size=num_left_right)
    left_outer_points = np.column_stack((left_outer_x, left_outer_y))
    points.extend(left_outer_points)

    # 在桌子右边外延区域生成点（x坐标大于桌子长度，y坐标在对应的范围内）
    right_outer_x = np.random.uniform(1.3, 1.4, size=num_left_right)
    right_outer_y = np.random.uniform(-0.825, 0.825, size=num_left_right)
    right_outer_points = np.column_stack((right_outer_x, right_outer_y))
    points.extend(right_outer_points)

    # 在桌子上边外延区域生成点（y坐标大于桌子宽度，x坐标在对应的范围内）
    top_outer_x = np.random.uniform(-1.3, 1.3, size=num_top_bottom)
    top_outer_y = np.random.uniform(0.725, 0.825, size=num_top_bottom)
    top_outer_points = np.column_stack((top_outer_x, top_outer_y))
    points.extend(top_outer_points)

    # 在桌子下边外延区域生成点（y坐标小于0，x坐标在对应的范围内）
    bottom_outer_x = np.random.uniform(-1.3, 1.3, size=num_top_bottom)
    bottom_outer_y = np.random.uniform(-0.725, -0.825, size=num_top_bottom)
    bottom_outer_points = np.column_stack((bottom_outer_x, bottom_outer_y))
    points.extend(bottom_outer_points)

    # 转换为numpy数组以便后续操作（可选，如果不需要对整体点集做统一的数组操作，可省略这步）
    points = np.array(points)
    np.random.shuffle(points)

    # # # # 以下是可视化部分，用于展示生成的点和桌子的位置关系，可按需注释掉
    # plt.plot([-1.4, 1.4, 1.4, -1.4, -1.4], [-0.825 -0.825, 0.825, 0.825, -0.825], 'r')  # 绘制桌子边框
    # plt.plot([-1, 1, 1, -1, -1], [-0.425, -0.425, 0.425, 0.425, -0.425], 'r')
    # plt.scatter(points[:, 0], points[:, 1])  # 绘制生成的点
    # plt.xlim(-1.5, 1.5)
    # plt.ylim(-1, 1)
    # plt.show()

    return points
    # print(points[0][0])

    # # # 以下是可视化部分，用于展示生成的点和桌子的位置关系，可按需注释掉
    # plt.plot([0, table_length, table_length, 0, 0], [0, 0, table_width, table_width, 0], 'r')  # 绘制桌子边框
    # plt.scatter(points[:, 0], points[:, 1])  # 绘制生成的点
    # plt.xlim(-outer_range, table_length + outer_range)
    # plt.ylim(-outer_range, table_width + outer_range)
    # plt.show()


if __name__ == "__main__":
    generrate_yuanxin(1000)
    
