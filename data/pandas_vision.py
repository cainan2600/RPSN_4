import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

# 文件名，可替换为实际使用的文件名
filename = "/home/cn/RPSN_4/work_dir/test24_MLP3_400epco_128hiden_1000train_400test_fk_ik_0.009ate_bz5_patience6_lossMSE45_shuffle_T_2relu_norm_atten/save_no_erro_data.txt"

# 用于存储解析后的数据
data = []
current_row = 0
line_num = 0
with open(filename, 'r') as file:
    lines = file.readlines()
    for line in lines:
        line_num += 1
        if not line_num == 11200:
            line = line.strip()
            if line:  # 只处理非空行
                parts = line.split(' ')
                x = float(parts[3])
                y = float(parts[4])
                data.append([x, y])
                current_row += 1
                if current_row == 7:  # 每处理7行数据后，下一行是空格行，直接跳过
                    current_row = 0
                    continue
        else:
            break


# 将数据转换为DataFrame格式
df = pd.DataFrame(data, columns=['x', 'y'])

# # 使用pandas绘制散点图
ax = df.plot.scatter(x='x', y='y', c='blue', label='Data Points')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Scatter Plot from 1400 data')
plt.plot([0, 4, 4, 0, 0], [0, 0, 2.6, 2.6, 0], 'r')
plt.plot([-0.4, 4.4, 4.4, -0.4, -0.4], [-0.4, -0.4, 3, 3, -0.4], 'r')


fig = ax.get_figure()
fig.savefig('/home/cn/RPSN_4/work_dir/test24_MLP3_400epco_128hiden_1000train_400test_fk_ik_0.009ate_bz5_patience6_lossMSE45_shuffle_T_2relu_norm_atten/save_no_erro_data.png')

plt.show()


