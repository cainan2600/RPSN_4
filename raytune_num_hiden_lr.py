import time
from torchviz import make_dot
import random
import numpy as np
# import matplotlib.pyplot as plt

import argparse
from torch.utils.data import Dataset, DataLoader, TensorDataset

from models import MLP_3, MLP_6, MLP_9, MLP_12
from lib.trans_all import *
from lib import IK, IK_loss, planner_loss
import torch
import torch.nn as nn
import math
import os
from lib.save import checkpoints
from lib.plot import *

import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray import train as ray_train

from ray.air.config import RunConfig
from ray.tune.schedulers import AsyncHyperBandScheduler

from torchviz import make_dot


class main():
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Training MLP")
        self.parser.add_argument('--batch_size', type=int, default=5, help='input batch size for training (default: 1)')
        # self.parser.add_argument('--learning_rate', type=float, default=0.002, help='learning rate (default: 0.003)')
        self.parser.add_argument('--epochs', type=int, default=400, help='gradient clip value (default: 300)')
        self.parser.add_argument('--clip', type=float, default=1, help='gradient clip value (default: 1)')
        self.parser.add_argument('--num_train', type=int, default=700)
        self.parser.add_argument('--num_test', type=int, default=300)
        self.args = self.parser.parse_args()

        # 使用cuda!!!!!!!!!!!!!!!未补齐
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 训练集数据导入
        self.load_train_data = torch.load('/home/cn/RPSN_4/data/data_cainan/5000-fk-ik-all-random-with-dipan/train/train_dataset_5000.pt')
        self.data_loader_train_dipan = torch.load("/home/cn/RPSN_4/data/data_cainan/5000-fk-ik-all-random-with-dipan/train/train_dataset_dipan_5000.pt")
        self.data_train = TensorDataset(self.load_train_data[:self.args.num_train], self.data_loader_train_dipan[:self.args.num_train])
        self.data_loader_train = DataLoader(self.data_train, batch_size=self.args.batch_size, shuffle=True)

        # 测试集数据导入
        self.load_test_data = torch.load('/home/cn/RPSN_4/data/data_cainan/5000-fk-all-random/test/test_dataset_400.pt')
        self.data_test = TensorDataset(self.load_test_data[:self.args.num_test])
        self.data_loader_test = DataLoader(self.data_test, batch_size=self.args.batch_size, shuffle=False)

        # 定义训练权重保存文件路径
        self.checkpoint_dir = r'/home/cn/RPSN_4/work_dir/test14_MLP3_400epco_128hiden_700train_300test_fk_0.002ate_lossMSE_train_all_random'
        # 多少伦保存一次
        self.num_epoch_save = 100

        # 选择模型及参数
        self.num_i = 6
        # self.num_h = config["num_h"]
        self.num_o = 3
        self.model = MLP_9
        
        # 如果是接着训练则输入前面的权重路径
        self.model_path = r''

        # 定义DH参数

        self.link_length = torch.tensor([0, -0.6127, -0.57155, 0, 0, 0])
        self.link_offset = torch.tensor([0.1807, 0, 0, 0.17415, 0.11985, 0.11655])
        self.link_twist = torch.FloatTensor([math.pi / 2, 0, 0, math.pi / 2, -math.pi / 2, 0])

    def train(self, config:dict):
        num_i = self.num_i
        num_h = config["num_h"]
        num_o = self.num_o

        epochs = self.args.epochs
        data_loader_train = self.data_loader_train
        learning_rate = config["lr"]
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
                data, lables = data
                inputs_bxxx6 = data
                num_zu_in_epoch = 0
                for inputs_xx6 in inputs_bxxx6:
                    num_zu_in_epoch += 1
                    # 将7x6打乱并转换为1x42
                    inputs_xx6 = inputs_xx6[torch.randperm(inputs_xx6.size(0))]
                    inputs = inputs_xx6


                    intermediate_outputs = model(inputs)
                    intermediate_outputs_list = intermediate_outputs.detach().numpy()
                    # print(intermediate_outputs.size())

                    # 得到每个1x6的旋转矩阵(7x6)
                    input_tar = shaping(inputs_xx6)           

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
                    IK_loss3 = torch.tensor(0.0, requires_grad=True)
                    loss_fn = torch.nn.MSELoss()

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

                            # 计算单IK_loss
                            IK_loss1, num_NOError1, num_NOError2 = IK_loss.calculate_IK_loss(angle_solution, the_NANLOSS_of_illegal_solution_with_num_and_Nan)
                            num_all_have_solution = num_all_have_solution - num_NOError1

                            # 总loss
                            IK_loss_batch = IK_loss_batch + IK_loss1
                        # else:
                        #     IK_loss1 = IK_loss1 + 0

                    # 不是每一组都有解即为失败
                    if num_all_have_solution == num_not_all_0:
                        NUM_all_have_solution += 1
                        IK_loss2 = IK_loss2 + 0
                    else:
                        IK_loss2 = IK_loss2 + loss_fn(outputs, lables[num_zu_in_epoch - 1]) * 400
                        # IK_loss2 = IK_loss2 + 1
                    IK_loss_batch = IK_loss_batch + IK_loss2
                    # print(IK_loss2)
                    
                    if 0<intermediate_outputs_list[1]<4:
                        if 0<intermediate_outputs_list[2]<2.6:
                            IK_loss3 = IK_loss3 + loss_fn(outputs, lables[num_zu_in_epoch - 1]) * 400
                        else:
                            IK_loss3 = IK_loss3 + 0
                    IK_loss_batch = IK_loss_batch + IK_loss3
                    
                    IK_loss_batch.retain_grad()

                    optimizer.zero_grad()  # 梯度初始化为零，把loss关于weight的导数变成0

                    # 定义总loss函数
                    loss = IK_loss_batch / len(input_tar)
                    loss.retain_grad()

                    loss.backward()  # 反向传播求梯度
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=self.args.clip)  # 进行梯度裁剪
                    optimizer.step()  # 更新所有梯度

            model.eval()

            data_loader_test = self.data_loader_test

            NUM_all_have_solution_test = 0

            for data_test in data_loader_test:
                with torch.no_grad():
                    inputs_bxxx6_test = data_test[0]
                    for inputs_xx6_test in inputs_bxxx6_test:
                        inputs_xx6_test = inputs_xx6_test[torch.randperm(inputs_xx6_test.size(0))]

                        inputs_test = inputs_xx6_test
                        intermediate_outputs_test = model(inputs_test)
                        input_tar_test = shaping(inputs_xx6_test)
                        outputs_test = torch.empty((0, 6))
                        pinjie1 = torch.cat([intermediate_outputs_test, torch.zeros(1).detach()])
                        pinjie2 = torch.cat([torch.zeros(2).detach(), pinjie1])
                        outputs_test = torch.cat([outputs_test, pinjie2.unsqueeze(0)], dim=0)

                        MLP_output_base_test = shaping(outputs_test)

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
                                # 计算IK_loss
                                # IK_loss_batch_test = IK_loss_batch_test + IK_loss_test1
                        if num_all_have_solution_test == num_not_all_0_test:
                            NUM_all_have_solution_test += 1
            accuracy = NUM_all_have_solution_test / self.args.num_test
            metrics_1={"acc":accuracy}

            ray_train.report(metrics_1)
                                    

if __name__ == "__main__":

    a = main()

    ray.init(num_cpus=16)

    search_space = {
        "num_h": tune.grid_search([64, 128]),
        "lr": tune.grid_search([0.0015, 0.003, 0.0045])
    }

    sched = AsyncHyperBandScheduler()

    tuner = tune.Tuner(
        # tune.with_resources(a.train, resources=ray.init()),
        tune.with_parameters(a.train),
        param_space=search_space,
        tune_config=tune.TuneConfig(
            metric="acc",
            mode="max",
            # scheduler=sched
        ),
        run_config=RunConfig(
            name="TuneTest_3loss300_have_L",
            storage_path="/home/cn/RPSN_4/work_dir/raytune_num_hiden_lr/mlp9",
            stop={
                "acc": 0.8
            }
        )
    )

    results = tuner.fit()
    
