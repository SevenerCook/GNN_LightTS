import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import math
from math import sqrt
from utils.masking import TriangularCausalMask, ProbMask
import os
from models.simple_linear import simple_linear
from layers.distribution_block import distribution_block
from torch.nn.parameter import Parameter

#移动平均模块，用于突出时间序列的趋势
class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        #为什么要进行前填充和后填充？
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)  #前填充
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)    #后填充
        x = torch.cat([front, x, end], dim=1)                   #拼接填充后的序列
        x = self.avg(x.permute(0, 2, 1))                                #进行1D平均池化
        x = x.permute(0, 2, 1)                                          #恢复原始维度
        return x

#序列分解模块，将时间序列分解为趋势和残差
class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)    #计算移动平均
        res = x - moving_mean               #计算残差
        return res, moving_mean             #返回残差和趋势

#图卷积网络模块
class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)    #第一层图卷积
        self.gc2 = GraphConvolution(nhid, nclass)   #第二层图卷积
        self.dropout = dropout

    def forward(self, x, adj):
        """
        前向传播
        :param x:batch_size, seq_len, num_features
        """
        x = F.relu(self.gc1(x, adj))                            #第一层图卷积+ReLU激活
        x = F.dropout(x, self.dropout, training=self.training)  #Dropout
        x = self.gc2(x, adj)                                    #第二层图卷积
        return F.log_softmax(x, dim=1)                          #返回log_softmax输出

#图卷积层
"""
基础层
改变运算法则
torch.einsum("bsl,le->bse", inputs, self.weight)->"bdsl,le->bdse"
output = torch.einsum("bsl,ble->bse", adj, support)
"""
class GraphConvolution(nn.Module):  # GCN AHW
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))   #权重矩阵(dim,dim)
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))              #偏置
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)                          #初始化权重
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)                        #初始化偏置

    def forward(self, inputs, adj):
        #inputs = inputs.transpose(1, 2)
        support = torch.einsum("bsl,le->bse", inputs, self.weight)#执行三维图卷积，创造邻接矩阵？
        output = torch.einsum("bsl,ble->bse", adj, support)       #图卷积操作
        if self.bias is not None:
            return output + self.bias                                   #加上偏置
        else:
            return output

#多图模块，用于处理多头注意力机制
class MultiGraph(nn.Module):

    def __init__(self, dim, d_model, n_heads, seq_len, pred_len, dropout=0.1):
        super(MultiGraph, self).__init__()

        self.n_heads = n_heads
        self.d_model = d_model
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.projection = nn.Linear(dim, n_heads * d_model)                     #投影层
        self.out_projection = nn.Linear(n_heads * d_model, dim)                 #输出投影层
        self.simple_linear = simple_linear(input_dim=self.seq_len,
                                           output_dim=self.pred_len)#线性层
        self.dropout = nn.Dropout(dropout)
        self.decompose = series_decomp(self.seq_len // 2 + 1 + self.seq_len // 4)#序列分解

    def forward(self, input):
        B, L, _ = input.shape   #B：batch_size, L:sequence_length
        H = self.n_heads
        D = self.d_model

        multi_features = self.projection(input).view(B, L, H, -1).permute(0, 2, 1, 3).contiguous().view(B, H, -1)
        att = torch.einsum("bsl,bhl->bsh", multi_features, multi_features)  #计算注意力分数
        att = self.dropout(torch.softmax(att, dim=-1))  #注意力权重

        new_multi_features = torch.tensor([]).to(input.device)
        for i in range(H):
            temp = torch.zeros_like(multi_features[:, 0, :]).view(B, 1, -1).float()
            for j in range(H):
                temp = temp + att[:, i, j].view(B, 1, 1) * multi_features[:, j, :].view(B, 1, -1)   #分别取不同通道的信号进行多头注意力机制
            new_multi_features = torch.cat((new_multi_features, temp), 1)
        new_multi_features = new_multi_features.view(B, L, -1)
        output = self.simple_linear(new_multi_features)
        output = self.out_projection(output)
        return output

#时间维度图模型
"""
总图层
input = input.unsqueeze(1)
"""
class TimeDimGraphModel(nn.Module):

    def __init__(self, configs):
        super(TimeDimGraphModel, self).__init__()
        self.configs = configs
        self.head_num = configs.head_num
        self.TimeDimGraphLayer_list = nn.ModuleList([])
        self.linear_list = nn.ModuleList([])
        for i in range(self.head_num):
            self.linear_list.append(nn.Linear(configs.c_out, configs.d_model))      #线性层(放在第一次数据处理前改变数据的feature),configs.c_out, configs.d_model
            self.TimeDimGraphLayer_list.append(TimeDimGraphLayer(configs.seq_len, configs.pred_len, configs.d_model, configs.dropout, configs.layer_num, configs.block_size, configs.cover_size))#中间图层(channel)，configs.seq_len->configs.pred_len
        self.projection = nn.Linear(self.head_num*configs.d_model, configs.c_out)   #投影层(feature),head_num*configs.d_model->configs.c_out
        self.dropout = nn.Dropout(configs.dropout)
        # self.contrast_loss = Contrast()

    def forward(self, input):
        B, L, D = input.shape   #B: batch_size, L:sequence_length, D:feature_dimensions
        # input = input.unsqueeze(1)
        output_list = torch.tensor([]).to(input.device)
        temp_input_list = []
        #没有进行数据分割，只是将所有数据分head_num次处理
        for i in range(self.head_num):
            temp_input = self.dropout(F.relu(self.linear_list[i](input)))       #线性变换+ReLU+Drop
            temp_input_list.append(temp_input)
            temp_output = self.TimeDimGraphLayer_list[i](temp_input)            #时间维度图层，根据有几个头分为几个TimeDimGraphLayer
            output_list = torch.cat([output_list, temp_output], dim=-1)
        output = self.dropout(self.projection(output_list))                     #投影+Dropout
        return output

#时间维度图层
class TimeDimGraphLayer(nn.Module):

    def __init__(self, input_length, pred_length, dim, dropout, layer_num=1, block_size=24, cover_size=12):
        super(TimeDimGraphLayer, self).__init__()
        self.input_length = input_length
        self.dim = dim
        self.layer_num = layer_num
        self.block_size = block_size
        self.cover_size = cover_size
        self.dropout = nn.Dropout(dropout)

        self.pred_length = pred_length

        self.TimeDimGraphBlock = TimeDimGraphBlock(input_length=self.input_length, dim=dim, dropout=dropout, layer_num=layer_num, block_size=block_size, cover_size=cover_size)
        self.output_length = block_size * ((self.input_length - block_size) // (block_size - cover_size) + 1 - layer_num)
        # self.projection = nn.Conv1d(in_channels=self.output_length, out_channels=pred_length, kernel_size=5, stride=1, padding=2,
        #                             padding_mode='circular', bias=False)
        self.simple_linear = simple_linear(input_dim=self.output_length, output_dim=pred_length)
        self.linear = nn.Linear(self.output_length, pred_length)
        self.seasonal_linear = nn.Linear(self.output_length, pred_length)   #季节线性层
        self.trend_linear = nn.Linear(self.output_length, pred_length)      #趋势线性层

    def forward(self, input):
        B, L, D = input.shape   #B:batch_size, L:sequence length, D:feature dimension
        temp_output= self.TimeDimGraphBlock(input)  #时间维度图块
        output = self.simple_linear(temp_output)    #简单线性变换
        return output

#时间维度图块（实现三个分解）
class TimeDimGraphBlock(nn.Module):

    def __init__(self, input_length, dim, dropout, layer_num, block_size=24, cover_size=12):
        super(TimeDimGraphBlock, self).__init__()
        self.input_length = input_length
        self.dim = dim
        self.layer_num = layer_num
        self.block_size = block_size
        self.cover_size = cover_size
        self.residue_size = block_size - cover_size
        self.block_num = (input_length - block_size) // self.residue_size

        self.TimeGenerateGraph_module = nn.ModuleList([])
        self.DimGenerateGraph_module = nn.ModuleList([])
        self.BlockGenerateGraph_module = nn.ModuleList([])
        self.GCN_Time = nn.ModuleList([])
        self.GCN_Dim = nn.ModuleList([])
        self.GCN_Block = nn.ModuleList([])
        self.projection_list = nn.ModuleList([])
        for l in range(layer_num):
            self.TimeGenerateGraph_module.append(
                TimeGenerateGraph(block_size,dim, dropout))                 #时间图生成模块
            self.DimGenerateGraph_module.append(
                DimGenerateGraph(block_size, dim, dropout))                 #维度图生成模块
            self.BlockGenerateGraph_module.append(
                BlockGenerateGraph(block_size, dim, dropout))               #块图生成模块
            self.GCN_Time.append(GraphConvolution(dim, dim))                #时间图卷积
            self.GCN_Dim.append(GraphConvolution(block_size, block_size))   #维度图卷积
            self.GCN_Block.append(GraphConvolution(dim, dim))               #块图卷积
            if l == 0:
                self.projection_list.append(
                    nn.Linear(self.input_length, block_size * ((self.input_length - block_size) // (block_size - cover_size) - l)))
            else:
                self.projection_list.append(nn.Linear(block_size * ((self.input_length - block_size) // (block_size - cover_size) + 1 - l), block_size * ((self.input_length - block_size) // (block_size - cover_size) - l)))

        self.dropout_Time = nn.Dropout()
        self.dropout_Dim = nn.Dropout()
        self.decomp = series_decomp(1 + block_size)                         #序列分解

    def forward(self, input):
        B, L, D = input.shape                           #B:batch_size, L:sequence length, D:feature dimensions
        for l in range(self.layer_num):
            output = torch.tensor([]).to(input.device)
            block_num = self.block_num - l
            for i in range(block_num):
                if l == 0:
                    ii = i * self.residue_size
                    temp_input1 = input[:, ii:ii + self.block_size, :]
                    temp_input2 = input[:, ii+ self.residue_size:ii + self.residue_size + self.block_size, :]
                else:
                    ii = i * self.block_size
                    temp_input1 = input[:, ii:ii + self.block_size, :]
                    temp_input2 = input[:, ii + self.block_size:ii + 2 * self.block_size, :]            #(不是与历史序列相关，而是变成未来序列相关？)

                TimeGraph = self.TimeGenerateGraph_module[l](temp_input1) # B, block_size, block_size,生成时间图
                DimGraph = self.DimGenerateGraph_module[l](temp_input1) # B, D, D，生成维度图
                BlockGraph = self.BlockGenerateGraph_module[l](temp_input1, temp_input2) # B, block_size, block_size

                BlockVector = self.GCN_Block[l](temp_input1, BlockGraph)  # B, block_size, D，块图卷积
                BlockVector = self.dropout_Time(F.relu(BlockVector)) + temp_input1

                TimeDimVector = self.GCN_Dim[l](BlockVector.permute(0, 2, 1), DimGraph)
                TimeDimVector = self.dropout_Dim(F.relu(TimeDimVector))
                TimeDimVector = TimeDimVector.permute(0, 2, 1) + BlockVector

                TimeDimBlockVector = self.GCN_Time[l](TimeDimVector, TimeGraph)
                TimeDimBlockVector = self.dropout_Time(F.relu(TimeDimBlockVector)) + TimeDimVector

                output = torch.cat([output, TimeDimBlockVector], dim=1)
            # trend_temp, seasonal_temp = self.decomp(trend_output)
            # seasonal_output = self.projection_list[l](seasonal.permute(0, 2, 1)).permute(0, 2, 1) + seasonal_temp
            input = output

        return output # B, L - block_size, D

#时间特征邻接矩阵生成模块
class TimeGenerateGraph(nn.Module):

    def __init__(self, input_length, dim, dropout):
        super(TimeGenerateGraph, self).__init__()
        self.softmax2d = nn.Softmax2d()
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Conv1d(in_channels=dim, out_channels=dim, kernel_size=5, stride=1, padding=2,
                                    padding_mode='circular', bias=False)
        # self.GCN = GraphConvolution(dim, dim)

    def forward(self, input):

        B, L, D = input.shape                                                                   #B: batch size, L: sequence length, D: feature dimension
        input = self.projection(input.permute(0, 2, 1)).transpose(1, 2)                         #1D卷积

        mean_input = input.mean(1, keepdim=True)
        std_input = torch.sqrt(torch.var(input, dim=1, keepdim=True, unbiased=False) + 1e-5)    #计算标准差

        input = (input - mean_input.repeat(1, L, 1)) / std_input                                #标准化
        # scale为None则为1./sqrt(E) 为true即有值，则为该值
        scale = 1. / sqrt(D)                                                                    #缩放因子
        # 内积 scores bhll
        #input = input.transpose(1, 2)

        scores = torch.einsum("ble,bse->bls", input, input)                               #计算内积
        cross_value = self.dropout(F.softmax((scale * scores), -3, _stacklevel=5))              #计算注意力权重

        # # GCN
        # TimeDimVector = self.GCN(input, cross_value)
        return cross_value # B, L, L

#维度图生成模块
class DimGenerateGraph(nn.Module):

    def __init__(self, input_length, dim, dropout):
        super(DimGenerateGraph, self).__init__()
        self.softmax2d = nn.Softmax2d()
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Conv1d(in_channels=input_length-1, out_channels=input_length-1, kernel_size=5, stride=1, padding=2,
                                    padding_mode='circular', bias=False)
        # self.GCN = GraphConvolution(dim, dim)

    def forward(self, input):
        input = torch.diff(input, dim=1)                            #计算差分
        B, L, D = input.shape
        input = self.projection(input).permute(0, 2, 1)             # B, D, L

        mean_input = input.mean(1, keepdim=True)                    #均值计算
        std_input = torch.sqrt(torch.var(input, dim=1,
                                         keepdim=True, unbiased=False) + 1e-5)  #计算标准差

        input = (input - mean_input.repeat(1, D, 1)) / std_input    #标准化
        scale = 1. / sqrt(L)                                        #缩放因子
        # 内积 scores bhll
        #input = input.transpose(1,2)

        scores = torch.einsum("ble,bse->bls", input, input)   #计算内积
        cross_value = self.dropout(F.softmax(
            (scale * scores), -3, _stacklevel=5))                   # B, D, D
        # # GCN
        # TimeDimVector = self.GCN(input, cross_value)
        return cross_value # B, D, D

#块图生成模块
class BlockGenerateGraph(nn.Module):

    def __init__(self, input_length, dim, dropout):
        super(BlockGenerateGraph, self).__init__()
        self.softmax2d = nn.Softmax2d()     #2D softmax归一化注意力分数
        self.dropout = nn.Dropout(dropout)  #Drop
        #第一个1D卷积层，处理输入1
        self.projection1 = nn.Conv1d(in_channels=dim, out_channels=dim, kernel_size=5, stride=1, padding=2,
                                    padding_mode='circular', bias=False)
        #第二个卷积层，处理输入2
        self.projection2 = nn.Conv1d(in_channels=dim, out_channels=dim, kernel_size=5, stride=1,
                                    padding=2,
                                    padding_mode='circular', bias=False)
        # self.GCN = GraphConvolution(dim, dim)

    def forward(self, input1, input2):

        B, L, D = input1.shape
        input1 = self.projection1(input1.permute(0, 2, 1)).permute(0, 2, 1) # B, L, D
        input2 = self.projection2(input2.permute(0, 2, 1)).permute(0, 2, 1)  # B, L, D

        mean_input1 = input1.mean(1, keepdim=True)
        std_input1 = torch.sqrt(torch.var(input1, dim=1, keepdim=True, unbiased=False) + 1e-5)
        mean_input2 = input2.mean(1, keepdim=True)
        std_input2 = torch.sqrt(torch.var(input2, dim=1, keepdim=True, unbiased=False) + 1e-5)

        input1 = (input1 - mean_input1.repeat(1, L, 1)) / std_input1
        input2 = (input2 - mean_input2.repeat(1, L, 1)) / std_input2

        # scale为None则为1./sqrt(E) 为true即有值，则为该值
        scale = 1. / sqrt(D)
        # 内积 scores bhll
        #input1 = input1.transpose(1, 2)
        #input2 = input2.transpose(1, 2)

        scores = torch.einsum("ble,bse->bls", input1, input2)
        cross_value = self.dropout(F.softmax((scale * scores), -3, _stacklevel=5))
        # # GCN
        # TimeDimVector = self.GCN(input, cross_value)
        return cross_value  # B, L, L


