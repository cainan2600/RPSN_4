7个输入，一个输出:7X6>>>>3x1

输入节点为6，并且将有效点位和无效点位打乱位置


一.训练

    cd ~/RPSN_2

    python run.py



二.数据集说明

    单个数据集为7x6，即七个样本，每个样本6个特征。每个数据集都是在同一个底盘位置时，先通过FK产生，再通过IK检验。

注：七个样本内只有1～7的随机个数的有效点位，无效点位都用[0， 0， 0， 0， 0， 0]代替


三.loss说明

    loss1：RPSN运动学损失

    loss2：一组中七个点位出现一个及以上错误，认为底盘位置预测错误，采用网络输出和实际底盘位置的mse损失

    loss3：预测出的底盘位置在桌子以内，即(0<x<4 and 0<y<2.6)，采用网络输出和实际底盘位置的mse损失


四.存在问题

1.对于是否符合运动学解的问题上呈现出过拟合，导致测试集很差：

    1.1 训练集准确率持续上涨，差不多到80%。测试集先上升后下降，且下降量很大。
具体结果参考文件：work_dir/test10_MLP3_new_600epco_1024hiden_1000data_fk_0.005rate_loss1_train_all_random

    1.2 增加数据集和添加dropout正则化后，效果不佳
具体结果参考文件：work_dir/test12_MLP3_new_600epco_1024hiden_2000data_fk_0.005ate_loss1_train_all_random

注：此时未添加loss2\loss3,主要分析运动学部分能力，使用MLP_9模型，9层隐藏层。

2.因因实际情况中，机器人不能到桌子里面，增加了loss2\loss3，解空间变小，在总体1000的测试集数据上准确率进一步降低，呈现出欠拟合，测试集还是上先上升后下降，且下降量很大

    2.1 loss2\loss3（乘以50倍，大小与loss1中IK解中异常跳出部分相同，与运动学loss1乘以1000倍相比很小）倍数太大，加上运动学过程中对于解存在NAN的惩罚过大，导致梯度爆炸
具体结果参考文件：work_dir/test14_MLP9_600epco_64hiden_1000train_400test_fk_ik_0.008ate_lossMSE50_train_all_random

    2.1 nan的loss放小到200倍，loss2\loss3放小到10倍（此时还是50倍时效果没有10倍好），有差不多500个点位错误，导致在总体1000的数据上准确率只能在60～65%左右。测试集上则更低：35～50%
具体结果参考文件：work_dir/test15_MLP9_400epco_64hiden_1000train_400test_fk_ik_0.005ate_lossMSE10_train_all_random


五.正确和错误的输入文件

    正确：work_dir/test15_MLP9_400epco_64hiden_1000train_400test_fk_ik_0.005ate_lossMSE10_train_all_random/save_no_erro_data
    错误：work_dir/test15_MLP9_400epco_64hiden_1000train_400test_fk_ik_0.005ate_lossMSE10_train_all_random/save_erro_data



