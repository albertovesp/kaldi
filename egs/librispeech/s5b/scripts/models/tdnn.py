#!/home/hzili1/tools/anaconda3/envs/py36/bin/python
# Copyright 2018 Yiwen Shao

# Apache 2.0

import torch
import torch.nn as nn
import torch.nn.functional as F

class tdnn_bn_relu(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, dilation=1):
        super(tdnn_bn_relu, self).__init__()
        self.tdnn = nn.Conv1d(in_dim, out_dim, kernel_size, dilation=dilation)
        self.bn = nn.BatchNorm1d(out_dim)

    def forward(self, x):
        assert len(x.size()) == 3  # x is of size (N, F, T)
        x = self.tdnn(x)
        x = F.relu(self.bn(x), inplace=True)
        return x

class tdnn_bn_relu_deconv(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, dilation=1):
        super(tdnn_bn_relu_deconv, self).__init__()
        self.tdnn = nn.ConvTranspose1d(in_dim, out_dim, kernel_size, dilation=dilation)
        self.bn = nn.BatchNorm1d(out_dim)

    def forward(self, x):
        assert len(x.size()) == 3  # x is of size (N, F, T)
        x = self.tdnn(x)
        x = F.relu(self.bn(x), inplace=True)
        return x

#class linear_bn_relu(nn.Module):
#    def __init__(self, in_dim, out_dim):
#        super(linear_bn_relu, self).__init__()
#        self.linear = nn.Linear(in_dim, out_dim)
#        self.bn = nn.BatchNorm1d(out_dim)
#        self.relu = nn.ReLU(inplace=True)
#
#    def forward(self, x):
#        assert len(x.size()) == 2  # x is of size (N, F, T)
#        x = self.linear(x)
#        x = self.bn(x)
#        x = self.relu(x)
#        return x

class TDNN(nn.Module):
    def __init__(self, in_dim, out_dim, num_layers, hidden_dims, kernel_sizes, dilations):
        super(TDNN, self).__init__()
        assert len(hidden_dims) == num_layers
        assert len(kernel_sizes) == num_layers
        assert len(dilations) == num_layers
        self.num_layers = num_layers
        self.tdnn_layers = self._make_layer(
            in_dim, hidden_dims, kernel_sizes, dilations)
        self.final_layer = nn.Conv1d(hidden_dims[-1], out_dim, kernel_size=1)

    def _make_layer(self, in_dim, hidden_dims, kernel_sizes, dilations):
        layers = []
        for i in range(self.num_layers):
            if i == 0:
                input_dim = in_dim
            else:
                input_dim = hidden_dims[i - 1]
            layers.append(tdnn_bn_relu(
                input_dim, hidden_dims[i], kernel_sizes[i], dilations[i]))

        return nn.Sequential(*layers)

    def forward(self, x):
        assert len(x.size()) == 3  # x is of size (T, N, F)
        # turn x to (N, F, T) for tdnn/cnn input
        x = x.transpose(0, 1).transpose(1, 2).contiguous()
        x = self.tdnn_layers(x)
        x = self.final_layer(x)
        x = x.transpose(2, 1).transpose(
            1, 0).contiguous()  # turn it back to (T, N, F)
        return x

class TDNN_SID(nn.Module):
    def __init__(self, in_dim, num_layers, hidden_dims, kernel_sizes, dilations):
        super(TDNN_SID, self).__init__()
        assert len(hidden_dims) == num_layers
        assert len(kernel_sizes) == num_layers
        assert len(dilations) == num_layers
        self.num_layers = num_layers
        self.tdnn_layers = self._make_layer(
            in_dim, hidden_dims, kernel_sizes, dilations)
        self.fc1 = nn.Linear(hidden_dims[-1] * 2, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 512)
        #self.bn2 = nn.BatchNorm1d(512)
        #self.output_layer = nn.Linear(512, out_dim)

    def _make_layer(self, in_dim, hidden_dims, kernel_sizes, dilations):
        layers = []
        for i in range(self.num_layers):
            if i == 0:
                input_dim = in_dim
            else:
                input_dim = hidden_dims[i - 1]
            layers.append(tdnn_bn_relu(
                input_dim, hidden_dims[i], kernel_sizes[i], dilations[i]))
        return nn.Sequential(*layers)

    #def _make_linear_layer(self, hidden_dims):
    #    layers = []
    #    layers.append(linear_bn_relu(hidden_dims[-1] * 2, 512))
    #    layers.append(linear_bn_relu(512, 512))
    #    return nn.Sequential(*layers)

    def forward(self, x):
        assert len(x.size()) == 3  # (N, T, F)    N for batch size, T for time, F for frequency
        x = x.transpose(1, 2).contiguous() # (N, F, T) 
        x = self.tdnn_layers(x) # (N, D1, T)
        mean_vec = torch.mean(x, 2) # (N, D1)
        std_vec = torch.std(x, 2) # (N, D1)
        x = torch.cat((mean_vec, std_vec), 1) # (N, 2D1)
        embedding_a = self.fc1(x) # (N, D2) 
        x = F.relu(self.bn1(embedding_a), inplace=True)
        embedding_b = self.fc2(x) 
        return embedding_a, embedding_b

class TDNN_SID_MULTI(nn.Module):
    def __init__(self, in_dim, num_layers, hidden_dims, kernel_sizes, dilations):
        super(TDNN_SID_MULTI, self).__init__()
        assert len(hidden_dims) == num_layers
        assert len(kernel_sizes) == num_layers
        assert len(dilations) == num_layers
        self.num_layers = num_layers
        self.tdnn_layers = self._make_layer(
            in_dim, hidden_dims, kernel_sizes, dilations)
        hidden_dims2 = list(reversed(hidden_dims))[1:]
        hidden_dims2.append(in_dim)
        self.tdnn_layers_deconv = self._make_layer(
            hidden_dims[-1], hidden_dims2, list(reversed(kernel_sizes)), list(reversed(dilations)), deconv=True)
        self.fc1 = nn.Linear(hidden_dims[-1] * 2, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 512)

    def _make_layer(self, in_dim, hidden_dims, kernel_sizes, dilations, deconv=False):
        layers = []
        for i in range(self.num_layers):
            if i == 0:
                input_dim = in_dim
            else:
                input_dim = hidden_dims[i - 1]
            if not deconv:
                layers.append(tdnn_bn_relu(
                  input_dim, hidden_dims[i], kernel_sizes[i], dilations[i]))
            else:
                layers.append(tdnn_bn_relu_deconv(
                  input_dim, hidden_dims[i], kernel_sizes[i], dilations[i]))
        return nn.Sequential(*layers)

    def forward(self, x):
        in_dim = x.size()
        assert len(x.size()) == 3  # (N, T, F)    N for batch size, T for time, F for frequency
        x = x.transpose(1, 2).contiguous() # (N, F, T) 
        x = self.tdnn_layers(x) # (N, D1, T)
        mean_vec = torch.mean(x, 2) # (N, D1)
        std_vec = torch.std(x, 2) # (N, D1)
        y = self.tdnn_layers_deconv(x).permute(0, 2, 1) # (N, T, F)
        out_dim = y.size()
        x = torch.cat((mean_vec, std_vec), 1) # (N, 2D1)
        embedding_a = self.fc1(x) # (N, D2) 
        x = F.relu(self.bn1(embedding_a), inplace=True)
        embedding_b = self.fc2(x)
        assert in_dim == out_dim
        return embedding_a, embedding_b, y

class TDNN_DIARIZATION(nn.Module):
    def __init__(self, in_dim, num_layers, hidden_dims, kernel_sizes, dilations):
        super(TDNN_DIARIZATION, self).__init__()
        assert len(hidden_dims) == num_layers
        assert len(kernel_sizes) == num_layers
        assert len(dilations) == num_layers
        self.num_layers = num_layers
        self.tdnn_layers = self._make_layer(
            in_dim, hidden_dims, kernel_sizes, dilations)
        self.fc1 = nn.Linear(hidden_dims[-1] * 2, 128)
        self.bn1 = nn.BatchNorm1d(128)

    def _make_layer(self, in_dim, hidden_dims, kernel_sizes, dilations):
        layers = []
        for i in range(self.num_layers):
            if i == 0:
                input_dim = in_dim
            else:
                input_dim = hidden_dims[i - 1]
            layers.append(tdnn_bn_relu(
                input_dim, hidden_dims[i], kernel_sizes[i], dilations[i]))
        return nn.Sequential(*layers)

    #def _make_linear_layer(self, hidden_dims):
    #    layers = []
    #    layers.append(linear_bn_relu(hidden_dims[-1] * 2, 512))
    #    layers.append(linear_bn_relu(512, 512))
    #    return nn.Sequential(*layers)

    def forward(self, x):
        assert len(x.size()) == 3  # (N, T, F)    N for batch size, T for time, F for frequency
        x = x.transpose(1, 2).contiguous() # (N, F, T) 
        x = self.tdnn_layers(x) # (N, D1, T)
        mean_vec = torch.mean(x, 2) # (N, D1)
        std_vec = torch.std(x, 2) # (N, D1)
        x = torch.cat((mean_vec, std_vec), 1) # (N, 2D1)
        embedding_a = self.fc1(x) # (N, D2) 
        x = self.bn1(embedding_a)
        return embedding_a, x 

def tdnn_sid_xvector(input_dim = 23):
    model = TDNN_SID(input_dim, 5, [512, 512, 512, 512, 1500], [5, 3, 3, 1, 1], [1, 2, 3, 1, 1]) 
    return model

def tdnn_sid_xvector_multi(input_dim = 40):
    model = TDNN_SID_MULTI(input_dim, 5, [512, 512, 512, 512, 1500], [5, 3, 3, 1, 1], [1, 2, 3, 1, 1])
    return model

def tdnn_sid_xvector_1(input_dim = 23):
    model = TDNN_SID(input_dim, 5, [512, 512, 512, 512, 512], [5, 3, 3, 1, 1], [1, 2, 3, 1, 1])
    return model

def tdnn_sid_xvector_2(input_dim = 23):
    model = TDNN_DIARIZATION(input_dim, 5, [512, 512, 512, 512, 1500], [5, 3, 3, 1, 1], [1, 2, 3, 1, 1])
    return model

if __name__ == "__main__":
    #kernel_size = 3
    #dilation = 2
    #num_layers = 1
    #hidden_dims = [20]
    #kernel_sizes = [3]
    #dilations = [2]
    #in_dim = 10
    #out_dim = 5
    in_dim = 23
    out_dim = 5139
    num_layers = 5
    hidden_dims = [512, 512, 512, 512, 1500]
    kernel_sizes = [5, 3, 3, 1, 1]
    dilations = [1, 2, 3, 1, 1]
    net = TDNN_SID(in_dim, out_dim, num_layers, hidden_dims, kernel_sizes, dilations)
    input = torch.randn(32, 217, 23)
    output = net(input)
    print(output[0].size())
    print(output[1].size())
