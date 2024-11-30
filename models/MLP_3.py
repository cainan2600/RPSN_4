import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP_self(nn.Module):
    def __init__(self, num_i, num_h, num_o, num_heads):
        super(MLP_self, self).__init__()

        self.attention = nn.MultiheadAttention(embed_dim=num_i, num_heads=num_heads, batch_first=True)  

        self.linear1 = torch.nn.Linear(num_i, num_h) 
        self.relu1 = torch.nn.ReLU()     
        self.tanh1 = torch.nn.Tanh()   

        # MLP 层
        self.linear2 = torch.nn.Linear(num_h, num_h) 
        self.relu2 = torch.nn.ReLU()
        self.tanh2 = torch.nn.Tanh() 

        # self.linear4 = torch.nn.Linear(num_h, num_h) 
        # self.relu4 = torch.nn.ReLU()

        self.linear3 = torch.nn.Linear(num_h, num_o) 
        self.dropout = torch.nn.Dropout(0.75)

    def forward(self, input):

        attn_output, _ = self.attention(input, input, input)
        input = input + attn_output

        x = self.linear1(input)
        x = self.relu1(x) 
        # x = self.tanh1(x)       


        x = self.linear2(x)
        # x = self.dropout(x)
        x = self.relu2(x)
        # x = self.tanh2(x)  

        # x = self.linear4(x)
        # x = self.dropout(x)
        # x = self.relu2(x)

        x = self.linear3(x)

        x = x.mean(dim=0)

        return x
