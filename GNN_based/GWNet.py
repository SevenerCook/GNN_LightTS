import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import sys

#实现图卷积操作
class nconv(nn.Module):
    def __init__(self):
        super(nconv,self).__init__()

    def forward(self,x, A):
        #x:batch_size,channels,num_nodes,seq_len
        #A:num_nodes, num_nodes 邻接矩阵
        #输出：batch_size, channels, num_nodes, seq_len
        x = torch.einsum('ncvl,vw->ncwl',(x,A))#可认为是特征的一次转化操作
        return x.contiguous()

class linear(nn.Module):
    """
    线性变换层
    #param c_in:通道数
    #param c_out:输出通道数
    """
    def __init__(self,c_in,c_out):
        super(linear,self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=True)

    def forward(self,x):
        """
        前向传播
        :param x: 输入特征，形状为(batch_size, c_in, num_nodes, seq_len)
        :return: 变换后的特征，形状为(batch_size, c_in, num_nodes, seq_len)
        """
        return self.mlp(x)
#实现图卷积网络
class gcn(nn.Module):
    def __init__(self,c_in,c_out,dropout,support_len=3,order=2):
        """
        图卷积网络
        :param c_in: 输入通道数
        :param c_out:输出通道数
        :param dropout: Dropout rate
        :param support_len: 邻接矩阵的数量
        :param order: 卷积的阶数
        """
        super(gcn,self).__init__()
        self.nconv = nconv()
        c_in = (order*support_len+1)*c_in#输入通道数的调整，2×
        self.mlp = linear(c_in,c_out)
        self.dropout = dropout
        self.order = order

    def forward(self,x,support):
        """
        前向传播
        :param x:输入特征，形状为(batch_size, c_in, num_nodes, seq_len)
        :param support: 邻接矩阵列表（无向图的邻接矩阵只有一个，除非是有向图有向后传播的邻接矩阵列表）
        :return: 图卷积后的特征，形状为(batch_size, c_out, num_nodes, seq_len)
        """
        out = [x]   #初始化输出列表
        for a in support:
            x1 = self.nconv(x,a)#一阶图卷积
            out.append(x1)
            for k in range(2, self.order + 1):#（此处仅使用邻接矩阵（无向图）进行高阶图卷积的计算，因为在高阶图卷积的过程中，邻接关系不变）
                x2 = self.nconv(x1,a)#高阶图卷积：高阶图卷积（Higher-Order Graph Convolution）是指在图卷积操作中，不仅考虑节点的一阶邻居（直接相连的节点），还考虑节点的多阶邻居（通过多条边相连的节点）。这种操作可以捕捉图中更远距离的依赖关系，从而更好地建模图的全局结构
                out.append(x2)
                x1 = x2

        h = torch.cat(out,dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h


class Model(nn.Module):
    def __init__(self, args):
        """
        主模型
        , device, num_nodes, dropout=0.3, supports=None, gcn_bool=True, addaptadj=True, aptinit=None, in_dim=2,out_dim=12,residual_channels=32,dilation_channels=32,skip_channels=256,end_channels=512,kernel_size=2,blocks=4,layers=2
        :param device: 设备选择（cpu/gpu）
        :param num_nodes:节点数量
        :param dropout: Dropout rate
        :param supports: 邻接矩阵列表
        :param gcn_bool: 是否使用图卷积
        :param addaptadj:是否使用自适应邻接矩阵
        :param aptinit:自适应邻接矩阵的初始化值
        :param in_dim:输入特征维度
        :param out_dim:输出特征维度
        :param residual_channels:残差连接的通道数
        :param dilation_channels:空洞卷积的通道数
        :param skip_channels:跳跃连接的通道数
        :param end_channels:最终输出的通道数
        :param kernel_size:卷积核大小
        :param blocks:WaveNet 块的数量
        :param layers:每个块中的层数
        """
        super(Model, self).__init__()
        self.dropout = 0.3
        self.blocks = 4
        self.layers = 2
        self.gcn_bool = True
        self.addaptadj = True
        #定义模块列表
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.gconv = nn.ModuleList()
        #初始卷积层
        self.start_conv = nn.Conv2d(in_channels=1,
                                    out_channels=32,
                                    kernel_size=(1,1))
        supports = None
        self.supports = supports
        aptinit = None
        addaptadj = True
        gcn_bool = True
        receptive_field = 1#初始化感受野
        self.supports_len = 0#初始化邻接矩阵列表长度
        if supports is not None:
            self.supports_len += len(supports)
        #自适应邻接矩阵
        if gcn_bool and addaptadj:
            if aptinit is None:
                if supports is None:
                    self.supports = []#supports不再是None
                #随机初始化自适应邻接矩阵的参数
                self.nodevec1 = nn.Parameter(torch.randn(args.num_nodes, 10).to(0), requires_grad=True).to(0)
                self.nodevec2 = nn.Parameter(torch.randn(10, args.num_nodes).to(0), requires_grad=True).to(0)
                self.supports_len +=1#无邻接矩阵的输入support_len=1
            else:
                if supports is None:
                    self.supports = []#supports不再是None
                # 使用 SVD 分解初始化自适应邻接矩阵的参数，获得对角矩阵的最大的前十个特征值，并转换成对角矩阵（10,10），得到对应于m的前十列线性变换矩阵(num_nodes,10)，
                m, p, n = torch.svd(aptinit)#aptinit是num_nodes, num_nodes
                initemb1 = torch.mm(m[:, :10], torch.diag(p[:10] ** 0.5))
                initemb2 = torch.mm(torch.diag(p[:10] ** 0.5), n[:, :10].t())#n是形状为（num_nodes, num_nodes）的矩阵，选取其前10列
                self.nodevec1 = nn.Parameter(initemb1, requires_grad=True).to(args.device)
                self.nodevec2 = nn.Parameter(initemb2, requires_grad=True).to(args.device)
                self.supports_len += 1#输入了邻接矩阵，进行稀疏化，得到support_len=1




        for b in range(4):
            additional_scope = 32 - 1
            new_dilation = 1
            for i in range(2):
                # dilated convolutions
                self.filter_convs.append(nn.Conv2d(in_channels=32,
                                                   out_channels=32,
                                                   kernel_size=(1,2),dilation=new_dilation))

                self.gate_convs.append(nn.Conv1d(in_channels=32,
                                                 out_channels=32,
                                                 kernel_size=(1, 2), dilation=new_dilation))

                # 1x1 convolution for residual connection
                self.residual_convs.append(nn.Conv1d(in_channels=32,
                                                     out_channels=32,
                                                     kernel_size=(1, 1)))

                # 1x1 convolution for skip connection
                self.skip_convs.append(nn.Conv1d(in_channels=32,
                                                 out_channels=256,
                                                 kernel_size=(1, 1)))
                #批归一化
                self.bn.append(nn.BatchNorm2d(32))
                new_dilation *=2
                receptive_field += additional_scope
                additional_scope *= 2
                #图卷积
                if self.gcn_bool:
                    self.gconv.append(gcn(32,32,self.dropout,support_len=self.supports_len))


        #输出层
        self.end_conv_1 = nn.Conv2d(in_channels=256,
                                  out_channels=512,
                                  kernel_size=(1,1),
                                  bias=True)

        self.end_conv_2 = nn.Conv2d(in_channels=512,
                                    out_channels=args.pred_len,
                                    kernel_size=(1,1),
                                    bias=True)

        self.receptive_field = receptive_field #最终感受野，感受野大小由模型确定



    def forward(self, input, batch_x_mark, dec_inp, batch_y_mark, batch_y):
        """
        前向传播
        :param input: 输入特征，形状为(batch_size, in_dim, num_nodes, seq_len)
        ：return : 输出特征，形状为(batch_size, out_dim, num_nodes, seq_len)
        """

        input = input.transpose(1, 2)
        input = input.unsqueeze(1)
        in_len = input.size(3)
        if in_len<self.receptive_field:
            x = nn.functional.pad(input,(self.receptive_field-in_len,0,0,0))
        else:
            x = input
        #初始卷积
        x = self.start_conv(x)
        skip = 0 #初始化跳跃连接

        # calculate the current adaptive adj matrix once per iteration 邻接矩阵的更新方法
        new_supports = None
        if self.gcn_bool and self.addaptadj and self.supports is not None:
            adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
            new_supports = self.supports + [adp] #仅有一个邻接矩阵

        # WaveNet layers
        for i in range(self.blocks * self.layers):

            #            |----------------------------------------|     *residual*
            #            |                                        |
            #            |    |-- conv -- tanh --|                |
            # -> dilate -|----|                  * ----|-- 1x1 -- + -->	*input*
            #                 |-- conv -- sigm --|     |
            #                                         1x1
            #                                          |
            # ---------------------------------------> + ------------->	*skip*

            #(dilation, init_dilation) = self.dilations[i]

            #residual = dilation_func(x, dilation, init_dilation, i)
            #残差连接
            residual = x
            # dilated convolution
            filter = self.filter_convs[i](residual)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](residual)
            gate = torch.sigmoid(gate)
            x = filter * gate

            # parametrized skip connection
            s = x
            s = self.skip_convs[i](s)
            try:
                skip = skip[:, :, :,  -s.size(3):]
            except:
                skip = 0
            skip = s + skip


            if self.gcn_bool and self.supports is not None:
                if self.addaptadj:
                    x = self.gconv[i](x, new_supports) #new_supports的内容仅有一个
                else:
                    x = self.gconv[i](x,self.supports)
            else:
                x = self.residual_convs[i](x)

            x = x + residual[:, :, :, -x.size(3):]


            x = self.bn[i](x)

        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        print("GWNet Xshape: ", x.shape)
        x = x.view(x.size(0), x.size(1), -1)

        return x





