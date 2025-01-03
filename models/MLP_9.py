import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP_self(nn.Module):
    def __init__(self, num_i, num_h, num_o):
        super(MLP_self, self).__init__()


        self.linear = torch.nn.Linear(num_i, num_h) # 6*64
        self.relu = torch.nn.ReLU()        

        # MLP 层
        self.linear1 = torch.nn.Linear(num_h, num_h*2) # 64x128
        self.relu1 = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(num_h*2, num_h*4) # 128x256
        self.relu2 = torch.nn.ReLU()
        self.linear3 = torch.nn.Linear(num_h*4, num_h*8) # 256x512
        self.relu3 = torch.nn.ReLU()
        self.linear4 = torch.nn.Linear(num_h*8, num_h*16) # 512x1024
        self.relu4 = torch.nn.ReLU()
        self.linear5 = torch.nn.Linear(num_h*16, num_h*8) # 1024x512
        self.relu5 = torch.nn.ReLU()
        self.linear6 = torch.nn.Linear(num_h*8, num_h*4) # 512*256
        self.relu6 = torch.nn.ReLU()
        self.linear7 = torch.nn.Linear(num_h*4, num_h*2) # 256x128
        self.relu7 = torch.nn.ReLU()
        self.linear8 = torch.nn.Linear(num_h*2, num_h) # 128*64
        self.relu8 = torch.nn.ReLU()

        self.linear9 = torch.nn.Linear(num_h, num_o) # 64x3
        # self.dropout = torch.nn.Dropout(0.5)

    def forward(self, input):

        # 通过注意力层
        # attention_out = self.attention(input)
        # x = self.relu(attention_out)

        x = self.linear(input)
        x = self.relu(x)        

        # 经过 MLP 层
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear3(x)
        x = self.relu3(x)
        x = self.linear4(x)
        x = self.relu4(x)
        x = self.linear5(x)
        x = self.relu5(x)
        x = self.linear6(x)
        x = self.relu6(x)
        x = self.linear7(x)
        x = self.relu7(x)
        x = self.linear8(x)
        x = self.relu8(x)
        x = self.linear9(x)

        x = x.mean(dim=0)

        return x
