import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP_self(nn.Module):
    def __init__(self, num_i, num_h, num_o):
        super(MLP_self, self).__init__()


        self.linear1 = torch.nn.Linear(num_i, num_h) # 6x64
        self.relu1 = torch.nn.ReLU()        

        # MLP 层
        self.linear2 = torch.nn.Linear(num_h, num_h*2) # 64*128
        self.relu2 = torch.nn.ReLU()
        self.linear4 = torch.nn.Linear(num_h*2, num_h*4) # 128*256
        self.relu4 = torch.nn.ReLU()
        self.linear5 = torch.nn.Linear(num_h*4, num_h*2) # 256*128
        self.relu5 = torch.nn.ReLU()
        self.linear6 = torch.nn.Linear(num_h*2, num_h) # 256*128
        self.relu6 = torch.nn.ReLU()

        self.linear3 = torch.nn.Linear(num_h, num_o) # 128*3
        self.dropout = torch.nn.Dropout(0.5)

    def forward(self, input):

        x = self.linear1(input)
        x = self.relu1(x)        


        x = self.linear2(x)
        x = self.relu2(x)
        # x = self.dropout(x)
        x = self.linear4(x)
        x = self.relu4(x)
        x = self.linear5(x)
        x = self.relu5(x)
        x = self.linear6(x)
        x = self.relu6(x)

        x = self.linear3(x)

        x = x.mean(dim=0)

        return x
