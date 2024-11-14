import torch
import torch.nn as nn

class MLP_self(nn.Module):

    def __init__(self, num_i, num_h, num_o):
        super(MLP_self, self).__init__()

        # self.mask1 = torch.FloatTensor([
        #     [1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        #     [0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        #     [0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        #     [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        #     [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0],
        #     [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0],
        #     [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1]
        # ])

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
        self.linear6 = torch.nn.Linear(num_h, num_h)
        self.relu6 = torch.nn.ReLU()
        self.linear7 = torch.nn.Linear(num_h, num_h)
        self.relu7 = torch.nn.ReLU()
        self.linear8 = torch.nn.Linear(num_h, num_h)
        self.relu8 = torch.nn.ReLU()
        self.linear9 = torch.nn.Linear(num_h, num_h)
        self.relu9 = torch.nn.ReLU()
        self.linear10 = torch.nn.Linear(num_h, num_h)
        self.relu10 = torch.nn.ReLU()
        self.linear11 = torch.nn.Linear(num_h, num_h)
        self.relu11 = torch.nn.ReLU()

        self.linear12 = torch.nn.Linear(num_h, num_h)
        self.relu12 = torch.nn.ReLU()
        self.linear13 = torch.nn.Linear(num_h, num_h)
        self.relu13 = torch.nn.ReLU()
        self.linear14 = torch.nn.Linear(num_h, num_h)
        self.relu14 = torch.nn.ReLU()
        self.linear15 = torch.nn.Linear(num_h, num_h)
        self.relu15 = torch.nn.ReLU()
        self.linear16 = torch.nn.Linear(num_h, num_h)
        self.relu16 = torch.nn.ReLU()
        self.linear17 = torch.nn.Linear(num_h, num_h)
        self.relu17 = torch.nn.ReLU()

        self.linear18 = torch.nn.Linear(num_h, num_o)
        self.dropout = torch.nn.Dropout(0.5)

    def forward(self, input):
        # input = self.mask1 * input
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
        x = self.relu6(x)
        x = self.linear7(x)
        x = self.relu7(x)
        x = self.linear8(x)
        x = self.relu8(x)
        x = self.linear9(x)
        x = self.relu9(x)
        x = self.linear10(x)
        x = self.relu10(x)
        x = self.linear11(x)
        x = self.relu11(x)

        x = self.linear12(x)
        x = self.relu12(x)
        x = self.linear13(x)
        x = self.relu13(x)
        x = self.linear14(x)
        x = self.relu14(x)
        x = self.linear15(x)
        x = self.relu15(x)
        x = self.linear16(x)
        x = self.relu16(x)
        x = self.linear17(x)
        x = self.relu17(x)

        x = self.linear18(x)
        return x