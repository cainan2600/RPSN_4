import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP_self(nn.Module):
    def __init__(self, num_i, num_h, num_o, num_heads):
        super(MLP_self, self).__init__()

        self.attention = nn.MultiheadAttention(embed_dim=num_i, num_heads=num_heads, batch_first=True)  
        self.batch_norm = nn.BatchNorm1d(num_features=num_i)

        self.linear1 = torch.nn.Linear(num_i, num_h) # 6x64
        self.relu1 = torch.nn.ReLU()        

        # MLP 层
        self.linear2 = torch.nn.Linear(num_h, num_h*2) # 64*128
        self.relu2 = torch.nn.ReLU()
        self.linear4 = torch.nn.Linear(num_h*2, num_h*4) # 128*256
        self.relu4 = torch.nn.ReLU()
        self.linear5 = torch.nn.Linear(num_h*4, num_h*2) # 256*128
        self.relu5 = torch.nn.ReLU()
        self.leak_relu1 = torch.nn.PReLU()
        self.linear6 = torch.nn.Linear(num_h*2, num_h) # 256*128
        self.relu6 = torch.nn.ReLU()
        self.leak_relu2 = torch.nn.PReLU()

        self.linear3 = torch.nn.Linear(num_h, num_o) # 128*3

        self.linear1.activation = 'relu'
        self.linear2.activation = 'relu'
        self.linear4.activation = 'relu'
        self.linear5.activation = 'relu'
        self.linear6.activation = 'relu'

        self.dropout = torch.nn.Dropout(0.5)
        self._initialize_weights()

    def forward(self, input):

        input = self.batch_norm(input)
        attn_output, _ = self.attention(input, input, input)
        input = input + attn_output

        x = self.linear1(input)
        x = self.relu1(x)        


        x = self.linear2(x)
        x = self.relu2(x)
        # x = self.dropout(x)
        x = self.linear4(x)
        x = self.relu4(x)
        x = self.linear5(x)
        x = self.relu5(x)
        # x = self.leak_relu1(x) 
        x = self.linear6(x)
        x = self.relu6(x)
        # x = self.leak_relu2(x) 

        x = self.linear3(x)

        x = x.mean(dim=0)

        return x

    def _initialize_weights(self):
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                if hasattr(layer, 'activation') and layer.activation == 'relu':
                    nn.init.kaiming_normal_(layer.weight, mode="fan_in", nonlinearity="relu")
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0)
                    # print("我HE初始化拉!!!!!!!!!", "{}".format(layer))
                else:
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0)
                    # print("我XA初始化拉!!!!!!!!!", "{}".format(layer))
