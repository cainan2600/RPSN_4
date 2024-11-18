import numpy as np
import math
 
 
def Link_Transformation(last_i,i,a_list,alpha_list,d_list,theta_list):
    """
    function：坐标系{i-1}到坐标系{i}的转换矩阵
    tips：这里的last_i指的是i-1
    """
    i = i  # 下面使用的i-1表示列表的第i-1个数，注意同DH参数里的i-1区别
    T_martix = np.mat(np.zeros((4,4)))
    
    T_martix[0,0] = np.cos(theta_list[i-1])
    T_martix[0,1] = -1*np.sin(theta_list[i-1])*np.cos(alpha_list[i-1])
    T_martix[0,2] = np.sin(theta_list[i-1])*np.sin(alpha_list[i-1])
    T_martix[0,3] = a_list[i-1]*np.cos(theta_list[i-1])
    
    T_martix[1,0] = np.sin(theta_list[i-1])
    T_martix[1,1] = np.cos(theta_list[i-1])*np.cos(alpha_list[i-1])
    T_martix[1,2] = -1*np.cos(theta_list[i-1])*np.sin(alpha_list[i-1])
    T_martix[1,3] = a_list[i-1]*np.sin(theta_list[i-1])
    
    T_martix[2,0] = 0
    T_martix[2,1] = np.sin(alpha_list[i-1])
    T_martix[2,2] = np.cos(alpha_list[i-1])
    T_martix[2,3] = d_list[i-1]
    
    T_martix[3,0] = 0
    T_martix[3,1] = 0 
    T_martix[3,2] = 0
    T_martix[3,3] = 1
    
    return T_martix
 
 
def get_zong_t(a, d, alpha, theta):
    a_list = a
    d_list = d
    alpha_list = alpha
    theta_list = theta
    
    T_0_1 = Link_Transformation(0,1,a_list,alpha_list,d_list,theta_list)
    # print(T_0_1)
    T_1_2 = Link_Transformation(1,2,a_list,alpha_list,d_list,theta_list)
    # print(T_1_2)
    T_2_3 = Link_Transformation(2,3,a_list,alpha_list,d_list,theta_list)
    # print(T_2_3)
    T_3_4 = Link_Transformation(3,4,a_list,alpha_list,d_list,theta_list)
    # print(T_3_4)
    T_4_5 = Link_Transformation(4,5,a_list,alpha_list,d_list,theta_list)
    # print(T_4_5)
    T_5_6 = Link_Transformation(5,6,a_list,alpha_list,d_list,theta_list)
    # print(T_5_6)
    
    T_0_6 = T_0_1*T_1_2*T_2_3*T_3_4*T_4_5*T_5_6
    # print(T_0_6)
    return T_0_6

if __name__ == "__main__":
    a_IK = [0, -0.6127, -0.57155, 0, 0, 0]
    d_IK = [0.1807, 0, 0, 0.17415, 0.11985, 0.11655] 
    alpha_IK = [math.pi / 2, 0, 0, math.pi / 2, -math.pi / 2, 0]

    theta = [0, 0, 0, 0, 0, 0]
    # for i in range(6):
    #     theta[i] = np.random.uniform(-np.pi, np.pi)

    TT = get_zong_t(a_IK, d_IK, alpha_IK, theta)



