import time
from torchviz import make_dot
import random
import numpy as np
# import matplotlib.pyplot as plt

import argparse
from torch.utils.data import Dataset, DataLoader, TensorDataset
from data.data_yanlong import train_dataset, test_dataset
from models import MLP_3, MLP_6, MLP_12, MLP_18
from lib.trans_all import *
from lib import IK, IK_loss, planner_loss
import torch
import torch.nn as nn
import math
import os
from lib.save import checkpoints
from lib.plot import *


class main():
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Training MLP")
        self.parser.add_argument('--batch_size', type=int, default=5, help='input batch size for training (default: 1)')
        self.parser.add_argument('--learning_rate', type=float, default=0.005, help='learning rate (default: 0.003)')
        self.parser.add_argument('--epochs', type=int, default=10, help='gradient clip value (default: 300)')
        self.parser.add_argument('--clip', type=float, default=1, help='gradient clip value (default: 1)')
        self.parser.add_argument('--num_train', type=int, default=2000)
        self.args = self.parser.parse_args()

        # 使用cuda!!!!!!!!!!!!!!!未补齐
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 训练集数据导入
        self.load_train_data = torch.load('/home/cn/RPSN_4/data/data_cainan/5000-fk-all-random/train/train_dataset_5000.pt')
        self.data_train = TensorDataset(self.load_train_data[:self.args.num_train])
        self.data_loader_train = DataLoader(self.data_train, batch_size=self.args.batch_size, shuffle=True)
        # 测试集数据导入
        self.load_test_data = torch.load('/home/cn/RPSN_4/data/data_cainan/5000-fk-all-random/test/test_dataset_400.pt')
        self.data_test = TensorDataset(self.load_test_data)
        self.data_loader_test = DataLoader(self.data_test, batch_size=self.args.batch_size, shuffle=False)

        # 定义训练权重保存文件路径
        self.checkpoint_dir = r'/home/cn/RPSN_4/work_dir/test12_MLP3_new_600epco_1024hiden_2000data_fk_0.005ate_loss1_train_all_random'
        # 多少伦保存一次
        self.num_epoch_save = 100

        # 选择模型及参数
        self.num_i = 6
        self.num_h = 128
        self.num_o = 3
        self.model = MLP_3
        
        # 如果是接着训练则输入前面的权重路径
        self.model_path = r''

        # 定义DH参数

        self.link_length = torch.tensor([0, -0.6127, -0.57155, 0, 0, 0])
        self.link_offset = torch.tensor([0.1807, 0, 0, 0.17415, 0.11985, 0.11655])
        self.link_twist = torch.FloatTensor([math.pi / 2, 0, 0, math.pi / 2, -math.pi / 2, 0])

    def train(self):
        num_i = self.num_i
        num_h = self.num_h
        num_o = self.num_o

        NUMError1 = []
        NUMError2 = []
        NUM_incorrect = []
        NUM_correct = []
        NUM_correct_test = []
        NUM_incorrect_test = []
        echo_loss = []
        echo_loss_test = []
        NUM_ALL_HAVE_SOLUTION = []
        NUM_ALL_HAVE_SOLUTION_test = []
        # NUM_2_to_1 = []
        # NUM_mid = []
        # NUM_lar = []
        # NUM_sametime_solution = []

        epochs = self.args.epochs
        data_loader_train = self.data_loader_train
        learning_rate = self.args.learning_rate
        model = self.model.MLP_self(num_i , num_h, num_o) 
        optimizer = torch.optim.Adagrad(model.parameters(), lr=learning_rate, weight_decay=0.000)  # 定义优化器
        # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.000)
        model_path = self.model_path

        if os.path.exists(model_path):          
            checkpoint = torch.load(model_path)  
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch'] + 1
            loss = checkpoint['loss']
            print('-' * 100 + '\n' + f"The loading model is complete, let's start this training from the {start_epoch} epoch, the current loss is : {loss}" + '\n' + '-' * 100)
        else:
            print('-' * 100 + '\n' + "There is no pre-trained model under the path, and the following training starts from [epoch1] after random initialization" + '\n' + '-' * 100)
            start_epoch = 1

        # 开始训练
        for epoch in range(start_epoch , start_epoch + epochs):
  
            sum_loss = 0.0
            sum_loss_test = 0.0
            numError1 = 0
            numError2 = 0
            num_incorrect = 0
            num_correct = 0
            NUM_all_have_solution = 0

            for data in data_loader_train:  # 读入数据开始训练
                inputs_bxxx6 = data[0]
                # 将batch_size中的每一组数据输入网络
                for inputs_xx6 in inputs_bxxx6:
                    # inputs = inputs_xx6
                    # 将7x6打乱并转换为1x42
                    inputs_xx6 = inputs_xx6[torch.randperm(inputs_xx6.size(0))]
                    # # print(inputs_xx6, inputs_xx6.size())
                    # inputs = shaping_inputs_xx6_to_1xx(inputs_xx6)
                    inputs = inputs_xx6


                    intermediate_outputs = model(inputs)
                    # print(intermediate_outputs.size())

                    # # 将1x42输入转为7x1x6,
                    # input_tar = shaping_inputs_1xx_to_xx1x6(inputs, num_i) # 得到变换矩阵

                    # 得到每个1x6的旋转矩阵(7x6)
                    input_tar = shaping(inputs_xx6)
                
                    # 将网络输出1x21转换为7x3
                    # intermediate_outputs = shaping_outputs_1xx_to_xx3(intermediate_outputs, num_i)

                    outputs = torch.empty((0, 6)) # 创建空张量
                    # for each_result in intermediate_outputs: # 取出每个batch_size中的每个数据经过网络后的结果1x3
                    pinjie1 = torch.cat([intermediate_outputs, torch.zeros(1).detach()])
                    pinjie2 = torch.cat([torch.zeros(2).detach(), pinjie1])
                    outputs = torch.cat([outputs, pinjie2.unsqueeze(0)], dim=0)

                    intermediate_outputs.retain_grad()
                    outputs.retain_grad()

                    MLP_output_base = shaping(outputs)  # 对输出做shaping运算-1X6变为4X4

                    MLP_output_base.retain_grad()

                    # 计算 IK_loss_batch
                    IK_loss_batch = torch.tensor(0.0, requires_grad=True)
                    IK_loss2 = torch.tensor(0.0, requires_grad=True)

                    num_all_have_solution = 0
                    num_not_all_0 = 0
                    for i in range(len(input_tar)):
                        if torch.all(inputs_xx6[i].ne(0)):
                            num_not_all_0 += 1
                            num_all_have_solution += 1
                            
                            angle_solution, num_Error1, num_Error2, the_NANLOSS_of_illegal_solution_with_num_and_Nan = IK.calculate_IK(
                                input_tar[i], 
                                MLP_output_base, 
                                self.link_length, 
                                self.link_offset, 
                                self.link_twist)

                            # 存在错误打印
                            numError1 = numError1 + num_Error1
                            numError2 = numError2 + num_Error2
                            # 计算单IK_loss
                            IK_loss1, num_NOError1, num_NOError2 = IK_loss.calculate_IK_loss(angle_solution, the_NANLOSS_of_illegal_solution_with_num_and_Nan)
                            num_all_have_solution = num_all_have_solution - num_NOError1

                            # 总loss
                            IK_loss_batch = IK_loss_batch + IK_loss1

                            # 有/无错误打印
                            num_incorrect = num_incorrect + num_NOError1
                            num_correct = num_correct + num_NOError2

                    # 不是每一组都有解即为失败
                    if num_all_have_solution == num_not_all_0:
                        NUM_all_have_solution += 1
                    else:
                        IK_loss2 = IK_loss2 + 10

                        IK_loss_batch = IK_loss_batch + IK_loss2
                    # print(IK_loss2)
                    IK_loss_batch.retain_grad()

                    optimizer.zero_grad()  # 梯度初始化为零，把loss关于weight的导数变成0

                    # 定义总loss函数
                    loss = IK_loss_batch
                    loss.retain_grad()

                    # 记录x轮以后网络模型checkpoint，用来查看数据流
                    if epoch % self.num_epoch_save == 0:
                        # print("第{}轮的网络模型被成功存下来了！储存内容包括网络状态、优化器状态、当前loss等".format(epoch))
                        checkpoints(model, epoch, optimizer, loss, self.checkpoint_dir)

                    loss.backward()  # 反向传播求梯度
                    # loss.backward(torch.ones_like(loss))  # 反向传播求梯度
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=self.args.clip)  # 进行梯度裁剪
                    optimizer.step()  # 更新所有梯度
                    sum_loss = sum_loss + loss.data

            echo_loss.append(sum_loss / (len(data_loader_train)))
            # print(echo_loss)
            
            NUMError1.append(numError1)
            NUMError2.append(numError2)
            NUM_incorrect.append(num_incorrect)
            NUM_correct.append(num_correct)
            NUM_ALL_HAVE_SOLUTION.append(NUM_all_have_solution)

            print("numError1", numError1)
            print("numError2", numError2)
            print("num_correct", num_correct)
            print("num_incorrect", num_incorrect)
            print('NUM_all_have_solution', NUM_all_have_solution)


            model.eval()

            data_loader_test = self.data_loader_test
            num_incorrect_test = 0
            num_correct_test = 0
            NUM_all_have_solution_test = 0
            # num_2_to_1 = 0
            # num_mid = 0
            # num_lar = 0
            # num_sametime_solution = 0
            # num_distance_large = 0
            for data_test in data_loader_test:
                with torch.no_grad():
                    inputs_bxxx6_test = data_test[0]
                    for inputs_xx6_test in inputs_bxxx6_test:
                        inputs_xx6_test = inputs_xx6_test[torch.randperm(inputs_xx6_test.size(0))]
                        # inputs_test = shaping_inputs_xx6_to_1xx(inputs_xx6_test)
                        inputs_test = inputs_xx6_test
                        intermediate_outputs_test = model(inputs_test)
                        # intermediate_outputs_test = intermediate_outputs_test.mean(dim=0)
                        # print(intermediate_outputs_test,intermediate_outputs_test.size())
                        input_tar_test = shaping(inputs_xx6_test)
                        outputs_test = torch.empty((0, 6))
                        pinjie1 = torch.cat([intermediate_outputs_test, torch.zeros(1).detach()])
                        pinjie2 = torch.cat([torch.zeros(2).detach(), pinjie1])
                        outputs_test = torch.cat([outputs_test, pinjie2.unsqueeze(0)], dim=0)

                        MLP_output_base_test = shaping(outputs_test)

                        # 计算 IK_loss_batch
                        IK_loss_batch_test = torch.tensor(0.0, requires_grad=True)
                        IK_loss3_test = torch.tensor(0.0, requires_grad=True)
                        num_all_have_solution_test = 0
                        num_not_all_0_test = 0                        
                        for i in range(len(input_tar_test)):
                            if torch.all(inputs_xx6_test[i].ne(0)):
                                num_not_all_0_test += 1
                                num_all_have_solution_test += 1                            
                                angle_solution = IK.calculate_IK_test(
                                    input_tar_test[i], 
                                    MLP_output_base_test, 
                                    self.link_length, 
                                    self.link_offset, 
                                    self.link_twist)
                                # IK时存在的错误打印
                                IK_loss_test1, IK_loss_test_incorrect, IK_loss_test_correct = IK_loss.calculate_IK_loss_test(angle_solution)
                                # 计算IK_loss时存在的错误与正确的打印
                                num_all_have_solution_test = num_all_have_solution_test - IK_loss_test_incorrect
                                num_incorrect_test = num_incorrect_test + IK_loss_test_incorrect
                                num_correct_test = num_correct_test + IK_loss_test_correct
                                # 计算IK_loss
                                # IK_loss_batch_test = IK_loss_batch_test + IK_loss_test1
                        if num_all_have_solution_test == num_not_all_0_test:
                            NUM_all_have_solution_test += 1                        

            print("num_correct_test", num_correct_test)
            print("num_incorrect_test", num_incorrect_test)
            print("NUM_all_have_solution_test", NUM_all_have_solution_test)
            # print('num_2_to1', num_2_to_1)
            # print('num_sametime_solution', num_sametime_solution)

            # NUM_2_to_1.append(num_2_to_1)
            # NUM_mid.append(num_mid)
            # NUM_lar.append(num_lar)
            NUM_incorrect_test.append(num_incorrect_test)
            NUM_correct_test.append(num_correct_test)
            NUM_ALL_HAVE_SOLUTION_test.append(NUM_all_have_solution_test)
            # NUM_sametime_solution.append(num_sametime_solution)

            print('[%d,%d] loss:%.03f' % (epoch, start_epoch + epochs-1, sum_loss / (len(data_loader_train))), "-" * 100)
            print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))

        # 画图
        plot_IK_solution(self.checkpoint_dir, start_epoch, epochs, len(self.data_test), NUM_incorrect_test, NUM_correct_test)
        plot_train(self.checkpoint_dir, start_epoch, epochs, self.args.num_train, NUMError1, NUMError2, NUM_incorrect, NUM_correct)
        plot_train_loss(self.checkpoint_dir, start_epoch, epochs, echo_loss)
        plot_no_not_have_solution(self.checkpoint_dir, start_epoch, epochs, NUM_ALL_HAVE_SOLUTION)
        plot_no_not_have_solution_test(self.checkpoint_dir, start_epoch, epochs, NUM_ALL_HAVE_SOLUTION_test)        # plot_test_loss(self.checkpoint_dir, start_epoch, epochs, echo_loss_test)
        # plot_2_to_1(self.checkpoint_dir, start_epoch, epochs, NUM_2_to_1, NUM_mid, NUM_lar)
        # plot_sametime_solution(self.checkpoint_dir, start_epoch, epochs, NUM_sametime_solution)

if __name__ == "__main__":
    a = main()
    a.train()
