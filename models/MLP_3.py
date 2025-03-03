import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP_self(nn.Module):
    def __init__(self, num_i, num_h, num_o, num_heads):
        super(MLP_self, self).__init__()

        self.attention1 = nn.MultiheadAttention(embed_dim=num_i, num_heads=num_heads, batch_first=True)
        self.batch_norm = nn.BatchNorm1d(num_features=num_i)

        self.linear1 = torch.nn.Linear(num_i, num_h)
        self.linear1.activation = 'relu'

        self.relu1 = torch.nn.ReLU()
        # self.tanh1 = torch.nn.Tanh()
        # self.leak_relu1 = torch.nn.PReLU()

        # MLP 层
        self.linear2 = torch.nn.Linear(num_h, num_h)
        self.linear2.activation = 'relu'

        self.relu2 = torch.nn.ReLU()
        # self.tanh2 = torch.nn.Tanh()
        # self.leak_relu2 = torch.nn.PReLU()

        # self.linear4 = torch.nn.Linear(num_h, num_h) 
        # self.relu4 = torch.nn.ReLU()

        self.linear3 = torch.nn.Linear(num_h, num_o)
        # self.dropout = torch.nn.Dropout(0.75)

        self._initialize_weights()

        # self.attention2 = nn.MultiheadAttention(embed_dim=num_o, num_heads=num_heads, batch_first=True) 

        self.register_buffer("mask", torch.tensor([0., 0., 1., 1., 1., 1.]))

    def forward(self, input):

        # input = self.batch_norm(input)
        # attn_output, _ = self.attention1(input, input, input)
        # input = input + attn_output

        input = input * self.mask

        x = self.linear1(input)
        x = self.relu1(x)
        # x = self.tanh1(x)
        # x = self.leak_relu1(x)       

        x = self.linear2(x)
        # x = self.dropout(x)
        x = self.relu2(x)
        # x = self.tanh2(x)
        # x = self.leak_relu2(x)  

        # x = self.linear4(x)
        # x = self.dropout(x)
        # x = self.relu2(x)

        x = self.linear3(x)

        # attn_output_out, _ = self.attention2(x, x, x)
        # x = x + attn_output_out

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