import torch
import torch.nn as nn
import torch.nn.functional as F
from openpyxl.styles.builtins import output
from GNN_based.net import Model2


class my_Layernorm(nn.Module):
    """
    Special designed layernorm for the seasonal part
    """
    def __init__(self, channels):
        super(my_Layernorm, self).__init__()
        self.layernorm = nn.LayerNorm(channels)

    def forward(self, x):
        """
        前向传播
        :param x:B,L,D
        :return:偏置后的的结果
        """
        x_hat = self.layernorm(x)
        bias = torch.mean(x_hat, dim=1).unsqueeze(1).repeat(1, x.shape[1], 1)
        return x_hat - bias


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        """
        前向传播
        :param x:B,L,D
        :return x:B,L,D
        """
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        """
        前向传播
        :param x:B,L,D
        :return res,moving_avg:(B,L,D), (B,L,D)
        """
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class EncoderLayer(nn.Module):
    """
    Autoformer encoder layer with the progressive decomposition architecture
    """
    def __init__(self, attention, d_model, d_ff=None, moving_avg=25, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model  #前馈网络的隐藏层维度
        self.attention = attention  #注意力机制
        self.conv1 = nn.ConvTranspose1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False, output_padding=0)
        self.conv2 = nn.ConvTranspose1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False, output_padding=0)
        self.decomp1 = series_decomp(moving_avg)
        self.decomp2 = series_decomp(moving_avg)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        """
        前向传播
        :param x:B,L,D
        :param attn_mask:B,L,D，注意力掩码
        :return res,attn:返回残差和注意力权重(B,L,D)，(B,L,L)
        """
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask
        )                               #注意力机制(B,L,D)，(B,L,D)
        x = x + self.dropout(new_x)     #残差连接+Drop
        x, _ = self.decomp1(x)          #序列分解
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))   #1D卷积+激活+Drop
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        res, _ = self.decomp2(x + y)
        return res, attn


class Encoder(nn.Module):
    """
    Autoformer encoder
    """
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer
        self.gnn = Model2()

    def forward(self, x, attn_mask=None):
        """
        前向传播
        :param x:B,L,D
        :param attn_mask:B,L,L,注意力掩码
        """


        attns = []                          #存储注意力机制权重
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x, attn = attn_layer(x, attn_mask=attn_mask)    #注意力层(B,L,D)，(B,L,L)
                x = conv_layer(x)                               #卷积层(B,L,D)
                x = self.gnn(x)
                attns.append(attn)                              #存储注意力权重
            x, attn = self.attn_layers[-1](x)                   #最后一层注意力(B,L,D)，(B,L,L)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)
        #x = self.gnn(x)
        return x, attns


class DecoderLayer(nn.Module):
    """
    Autoformer decoder layer with the progressive decomposition architecture
    """
    def __init__(self, self_attention, cross_attention, d_model, c_out, d_ff=None,
                 moving_avg=25, dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model                  #前馈网络的隐藏层维度
        self.self_attention = self_attention        #自注意力机制
        self.cross_attention = cross_attention      #交叉注意力机制
        self.conv1 = nn.ConvTranspose1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False)
        self.conv2 = nn.ConvTranspose1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False)
        self.decomp1 = series_decomp(moving_avg)    #序列分解
        self.decomp2 = series_decomp(moving_avg)
        self.decomp3 = series_decomp(moving_avg)
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=3, stride=1, padding=1,
                                    padding_mode='circular', bias=False)    #1D投影卷积层，什么是1维投影卷积层？
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        """
        前向传播
        :param x:B,L,D
        :param cross_mask:B,L,D,交叉输入？
        :param x_mask:B,L,L,自注意力掩码
        :param cross_mask:B,L,L 交叉注意力掩码
        :return x,residual_trend:(B,L,D),(B,L,c_out)返回解码结果和残差趋势
        """
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask
        )[0])                           #自注意力+残差链接+Dropout
        x, trend1 = self.decomp1(x)     #序列分解(B,L,D)，(B,L,D)
        x = x + self.dropout(self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask
        )[0])
        x, trend2 = self.decomp2(x)
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        x, trend3 = self.decomp3(x + y)

        residual_trend = trend1 + trend2 + trend3           #残差趋势(B,L,D)
        residual_trend = self.projection(
            residual_trend.permute(0, 2, 1)).transpose(1, 2)#投影(B,L,c_out)
        return x, residual_trend


class Decoder(nn.Module):
    """
    Autoformer encoder(decoder?)
    """
    def __init__(self, layers, norm_layer=None, projection=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None, trend=None):
        """
        前向传播
        :param x:B,L,D
        :param cross:(B,L,D),交叉输入
        :param x_mask:(B,L,L)自注意力掩码
        :param cross_mask:(B,L,L)交叉注意力掩码
        :param trend:(B,L,c_out)初始趋势
        :return x,trend:(B,L,D),(B,L,c_out)
        """
        for layer in self.layers:
            x, residual_trend = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)   #解码器层(B,L,D)，(B,L,c_out)
            trend = trend + residual_trend  #更新趋势(B,L,c_out)

        if self.norm is not None:
            x = self.norm(x)                #归一化(B,L,D)

        if self.projection is not None:
            x = self.projection(x)          #投影(B,L,c_out)
        return x, trend
