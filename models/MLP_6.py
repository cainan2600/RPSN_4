import torch
import torch.nn as nn

class MLP_self(nn.Module):

    def __init__(self, num_i, num_h, num_o):
        super(MLP_self, self).__init__()

        self.linear1 = torch.nn.Linear(num_i, num_h, bias=False)
        self.relu = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(num_h, num_h, bias=False)
        self.relu2 = torch.nn.ReLU()
        self.linear3 = torch.nn.Linear(num_h, num_h)
        self.relu3 = torch.nn.ReLU()
        self.linear4 = torch.nn.Linear(num_h, num_h)
        self.relu4 = torch.nn.ReLU()
        self.linear5 = torch.nn.Linear(num_h, num_h)
        self.relu5 = torch.nn.ReLU()
        self.linear6 = torch.nn.Linear(num_h, num_o)
        self.dropout = torch.nn.Dropout(0.5)

    def forward(self, input):
        x = self.linear1(input)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear3(x)
        x = self.relu3(x)
        x = self.linear4(x)
        x = self.relu4(x)
        x = self.linear5(x)
        x = self.relu5(x)
        x = self.linear6(x)
        return x