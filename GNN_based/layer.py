from __future__ import division
import torch
import torch.nn as nn
from torch.nn import init
import numbers
import torch.nn.functional as F

#定义图卷积操作类
class nconv(nn.Module):
    def __init__(self):
        super(nconv,self).__init__()

    def forward(self,x, A):
        """
        前向传播
        ：param x: （样本数量，通道数量，特征数量，时间序列步长）
        ：param A: （目标节点数，源节点数）
        :return x:样本数量，通道数量，目标节点数量，时间序列步长
        """
        x = torch.einsum('ncwl,vw->ncvl',(x,A)) #n(样本数量)x1（通道数量）x特征数量（输入节点的数量）x窗口大小（时间序列步长） mul v(目标节点数)xw(源节点的数量) =》1x1x目标节点数x时间序列步长
        return x.contiguous() #返回连续的内存块，便于后续高效操作

#定义动态卷积操作类（与非动态卷积的邻接矩阵的维度不一样？）
class dy_nconv(nn.Module):
    def __init__(self):
        super(dy_nconv,self).__init__()

    def forward(self,x, A):
        """
        前向传播
        :param x:（样本数量，通道数量，特征数量，时间序列步长）
        :param A:动态邻接矩阵(样本数量，目标节点数量，源节点数量，时间步长)
        :return x:样本数量x通道数量x目标节点数量x时间序列
        """
        x = torch.einsum('ncvl,nvwl->ncwl',(x,A))#ncvl:样本数量x通道数量x目标节点数量x时间序列步长 nvwl:样本数量x目标节点数量x源节点数量x时间步长
        return x.contiguous()

class linear(nn.Module):
    def __init__(self,c_in,c_out,bias=True):
        super(linear,self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=bias)

    def forward(self,x):
        return self.mlp(x)

#定义图卷积传播类，prop 类实现了一个图卷积传播模块，用于在图结构数据上进行信息传播。它通过多层图卷积操作，将节点的特征与其邻居节点的特征进行聚合，从而更新节点的表示。
class prop(nn.Module):
    def __init__(self,c_in,c_out,gdep,dropout,alpha):#图卷积的过程，gdep表示深度，alpha表示控制保留原始节点信息的比例
        """
        初始化结构函数
        :param c_in:输入通道数
        :param c_out:输出通道数
        :param gdep:图卷积的深度（控制层数？）
        :param alpha:控制原始节点信息的比例
        """
        super(prop, self).__init__()
        self.nconv = nconv()
        self.mlp = linear(c_in,c_out)
        self.gdep = gdep
        self.dropout = dropout
        self.alpha = alpha

    def forward(self,x,adj):
        """
        前向传播
        :param x:输入（样本数量,通道数量,特征数量,时间序列长度）
        :param adj:邻接矩阵
        :return ho:隐藏状态矩阵（？）
        """
        adj = adj + torch.eye(adj.size(0)).to(x.device)#添加自环，这个邻接矩阵是动态的吗（哪里来的）？
        d = adj.sum(1)#计算每个节点的度（为什么计算每个节点的度？用于归一化邻接矩阵）
        h = x#初始化隐藏状态
        dv = d#度矩阵（度矩阵又是干什么的？用于归一化邻接矩阵，与GWNet不同？？？）
        a = adj / dv.view(-1, 1) #归一化邻接矩阵
        for i in range(self.gdep):
            h = self.alpha*x + (1-self.alpha)*self.nconv(h,a)#更新gdep次h的值，x一直是每一层的外输入？？？
        ho = self.mlp(h)
        return ho

#定义混合传播类
class mixprop(nn.Module):
    def __init__(self,c_in,c_out,gdep,dropout,alpha):#混合传播过程
        """
        :param c_in:输入通道数
        :param c_out:输出通道数
        :param gdep:图卷积深度
        :param alpha:控制保留原始节点信息的比例
        """
        super(mixprop, self).__init__()
        self.nconv = nconv()
        self.mlp = linear((gdep+1)*c_in,c_out)
        self.gdep = gdep
        self.dropout = dropout
        self.alpha = alpha


    def forward(self,x,adj):
        """
        前向传播
        :param x:输入（样本...）
        :param adj:邻接矩阵（是否是动态？）
        :return ho:隐藏状态矩阵（？）
        """
        adj = adj + torch.eye(adj.size(0)).to(x.device)
        d = adj.sum(1)
        h = x
        out = [h]
        a = adj / d.view(-1, 1)
        for i in range(self.gdep):
            h = self.alpha*x + (1-self.alpha)*self.nconv(h,a)
            out.append(h)#保存了每一层传播的结果
        ho = torch.cat(out,dim=1)
        ho = self.mlp(ho)
        return ho
#动态混合传播类
class dy_mixprop(nn.Module):
    def __init__(self,c_in,c_out,gdep,dropout,alpha):#动态混合传播过程，动态生成邻接矩阵并进行传播
        """
        :param c_in:输入通道数
        :param c_out:输出通道数
        :param gdep:图卷积深度
        :param alpha:控制保留原始节点信息的比例
        """
        super(dy_mixprop, self).__init__()
        self.nconv = dy_nconv()
        self.mlp1 = linear((gdep+1)*c_in,c_out)
        self.mlp2 = linear((gdep+1)*c_in,c_out)

        self.gdep = gdep
        self.dropout = dropout
        self.alpha = alpha
        self.lin1 = linear(c_in,c_in)#用于生成动态邻接矩阵
        self.lin2 = linear(c_in,c_in)#用于生成动态邻接矩阵


    def forward(self,x):
        """
        前向传播(没有邻接矩阵？？？？)
        :param x:输入（样本数量，通道数量，特征数量，序列长度）
        :return:ho1+ho2（反向和正向传播的结果之和）
        """
        #adj = adj + torch.eye(adj.size(0)).to(x.device)
        #d = adj.sum(1)
        x1 = torch.tanh(self.lin1(x))#ncwl，动态生成邻接矩阵的间接过程，使用线性函数生成数据1（样本数量，通道数量，特征数量，序列长度）
        x2 = torch.tanh(self.lin2(x))#ncwl，动态生成邻接矩阵的间接过程，使用线性函数生成数据2（样本数量，通道数量，特征数量，序列长度）
        adj = self.nconv(x1.transpose(2,1),x2)#nvwl,动态生成邻接矩阵（x.T 为（样本数量，特征数量，通道数量，序列长度）×（样本数量，通道数量，特征数量，序列长度）），使用动态卷积获得动态邻接矩阵
        #adj:(样本数量，节点数，节点数，序列长度)
        #邻接矩阵归一化
        adj0 = torch.softmax(adj, dim=2)#归一化邻接矩阵
        adj1 = torch.softmax(adj.transpose(2,1), dim=2)#归一化邻接矩阵的转置

        #使用动态邻接矩阵进行信息传播，正向传播
        h = x#初始化隐藏状态（第一层的隐藏状态即是输入）
        out = [h] #保存每一层的输出
        for i in range(self.gdep):#进行多图层卷积过程
            h = self.alpha*x + (1-self.alpha)*self.nconv(h,adj0)#进行层图卷积过程，x一直是每一层的外输入？？？，是的，原始信息
            out.append(h)#保存下每个图卷积过程的状态
        #拼接不同深度的信息
        ho = torch.cat(out,dim=1)#在通道数上进行叠加
        #对拼接后的特征进行线性变换
        ho1 = self.mlp1(ho)

        #反向传播，因为使用的是相反的邻接矩阵
        h = x
        out = [h]
        for i in range(self.gdep):
            h = self.alpha * x + (1 - self.alpha) * self.nconv(h, adj1)#进行层图卷积过程，x一直是每一层的外输入？？？
            out.append(h)
        ho = torch.cat(out, dim=1)
        ho2 = self.mlp2(ho)

        return ho1+ho2


#空洞一维卷积
class dilated_1D(nn.Module):
    def __init__(self, cin, cout, dilation_factor=2):
        """
        空洞卷积
        :param cin:输入通道数
        :param cout:输出通道数
        :param dilation_factor:空洞因子
        """
        super(dilated_1D, self).__init__()
        self.tconv = nn.ModuleList()
        self.kernel_set = [2,3,6,7]
        self.tconv = nn.Conv2d(cin,cout,(1,7),dilation=(1,dilation_factor))

    def forward(self,input):
        """
        前向传播
        :param input: 输入特征，形状为（batch_size, cin, num_nodes, seq_len）
        :return x:输出特征(batch_size, cin, num_nodes, seq_len)
        """
        x = self.tconv(input)
        return x

class dilated_inception(nn.Module):
    def __init__(self, cin, cout, dilation_factor=2):
        """
        多项空洞卷积
        :param cin:输入通道数
        :param cout:输出通道数
        :param dilation_factor:空洞卷积因子
        """
        super(dilated_inception, self).__init__()
        self.tconv = nn.ModuleList()
        self.kernel_set = [2,3,6,7]
        cout = int(cout/len(self.kernel_set)) #每个卷积核的通道数
        for kern in self.kernel_set:
            self.tconv.append(nn.Conv2d(cin,cout,(1,kern),dilation=(1,dilation_factor)))

    def forward(self,input):
        """
        前向传播
        :param input:输入，batch_size, cin, num_nodes, seq_len
        :return x:形状为（batch_size, cout, num_nodes, seq_len）
        """
        x = []
        for i in range(len(self.kernel_set)):#使用每个卷积核进行卷积操作
            x.append(self.tconv[i](input))
        for i in range(len(self.kernel_set)):#对每个卷积核进行序列长度的裁剪，进行对齐
            x[i] = x[i][...,-x[-1].size(3):]
        x = torch.cat(x,dim=1) #拼接不同卷积核的结果，形状为（batch_size, cout, num_nodes, seq_len）
        return x

#从数据中学习并生成邻接矩阵。它通过节点嵌入（node embeddings）或静态特征（static features）来计算节点之间的相似度，从而构建图的邻接矩阵。
class graph_constructor(nn.Module):
    def __init__(self, nnodes, k, dim, device, alpha=3, static_feat=None):#生成邻接矩阵
        """
        构建图的邻接矩阵
        :param nnodes:节点数量
        :param k:每个节点的top_K邻居(需要手动修改)
        :pram dim:输出维度
        :param alpha:控制激活函数的斜率，为什么要控制斜率？？？？
        :param static_feat:静态特征
        """
        super(graph_constructor, self).__init__()
        self.nnodes = nnodes
        if static_feat is not None:#static_feat表示节点的额外属性
            xd = static_feat.shape[1]
            self.lin1 = nn.Linear(xd, dim)
            self.lin2 = nn.Linear(xd, dim)
        else:
            self.emb1 = nn.Embedding(nnodes, dim)
            self.emb2 = nn.Embedding(nnodes, dim)
            self.lin1 = nn.Linear(dim,dim)
            self.lin2 = nn.Linear(dim,dim)

        self.device = device
        self.k = k
        self.dim = dim
        self.alpha = alpha
        self.static_feat = static_feat

    def forward(self, idx):
        """
        前向传播
        :param idx: 节点序号
        :return :（num_nodes, num_nodes）
        """
        #检查是否有静态特征，有则用对应idx的，无则使用节点嵌入构建节点向量
        if self.static_feat is None:
            nodevec1 = self.emb1(idx)
            nodevec2 = self.emb2(idx)
        else:
            nodevec1 = self.static_feat[idx,:]
            nodevec2 = nodevec1
        nodevec1 = torch.tanh(self.alpha*self.lin1(nodevec1))#
        nodevec2 = torch.tanh(self.alpha*self.lin2(nodevec2))

        a = torch.mm(nodevec1, nodevec2.transpose(1,0))-torch.mm(nodevec2, nodevec1.transpose(1,0))#通过矩阵乘法计算节点之间的相似度
        adj = F.tanh(torch.tanh(self.alpha*a))
        mask = torch.zeros(idx.size(0), idx.size(0)).to(0)#初始化掩码，这里为什么需要掩码？？？
        mask.fill_(float('0'))#填充掩码
        s1,t1 = (adj + torch.rand_like(adj)*0.01).topk(self.k,1)#只保留每个节点的 top-k 邻居，生成稀疏邻接矩阵
        mask.scatter_(1,t1,s1.fill_(1))#应用掩码
        adj = adj*mask
        return adj

    def fullA(self, idx):
        """
        生成邻接矩阵
        :param idx:输入特定的节点
        return adj:生成的邻接矩阵
        """
        if self.static_feat is None:
            nodevec1 = self.emb1(idx)
            nodevec2 = self.emb2(idx)
        else:
            nodevec1 = self.static_feat[idx,:]
            nodevec2 = nodevec1

        nodevec1 = torch.tanh(self.alpha*self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha*self.lin2(nodevec2))

        a = torch.mm(nodevec1, nodevec2.transpose(1,0))-torch.mm(nodevec2, nodevec1.transpose(1,0))
        adj = F.relu(torch.tanh(self.alpha*a))
        return adj

class graph_global(nn.Module):
    def __init__(self, nnodes, k, dim, device, alpha=3, static_feat=None):
        """
        全局图的构造类（什么是全局图？？？）
        :param nnodes:节点数量
        :param k:每个节点的top-k邻居
        :param dim:嵌入维度
        :param device:设备选择
        :param alpha:控制激活函数的斜率
        :param static_feat：静态特征（什么是静态特征）
        """
        super(graph_global, self).__init__()
        self.nnodes = nnodes
        self.A = nn.Parameter(torch.randn(nnodes, nnodes).to(device), requires_grad=True).to(device)#全局邻接矩阵（为什么又来一个全局邻接矩阵？？？）

    def forward(self, idx):
        """
        前向传播
        :param idx: 节点索引，形状（num_nodes,）没用到为什么还要传输？？？
        :return:全局邻接矩阵，形状为(num_nodes,num_nodes)
        """
        return F.relu(self.A)


class graph_undirected(nn.Module):
    def __init__(self, nnodes, k, dim, device, alpha=3, static_feat=None):
        """
        无向图构造类
        :param nnodes:节点数量
        :param k:每个节点的top-k邻居
        :param dim:嵌入维度（什么是嵌入维度？？？）
        :param device:设备
        :param alpha:控制激活函数的斜率
        :param static_feat:静态特征（静态特征的作用？？？？），（num_nodes, num_features）
        """
        super(graph_undirected, self).__init__()
        self.nnodes = nnodes
        if static_feat is not None:#如果有静态特征
            xd = static_feat.shape[1]
            self.lin1 = nn.Linear(xd, dim)#线性变换
        else:
            self.emb1 = nn.Embedding(nnodes, dim) #节点嵌入
            self.lin1 = nn.Linear(dim,dim) #线性变换

        self.device = device
        self.k = k  #每个节点的top_k邻居
        self.dim = dim  #嵌入维度
        self.alpha = alpha  #控制激活函数的斜率
        self.static_feat = static_feat  #静态特征

    def forward(self, idx):
        """
        前向传播
        :param idx:节点索引，形状为（num_nodes, ）
        :return: 稀疏邻接矩阵，形状为(num_nodes,num_nodes)
        """
        if self.static_feat is None:
            nodevec1 = self.emb1(idx)
            nodevec2 = self.emb1(idx)
        else:
            nodevec1 = self.static_feat[idx,:]#提取对应节点的的特征
            nodevec2 = nodevec1

        nodevec1 = torch.tanh(self.alpha*self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha*self.lin1(nodevec2))

        a = torch.mm(nodevec1, nodevec2.transpose(1,0)) #构建初始邻接矩阵
        adj = F.relu(torch.tanh(self.alpha*a))  #激活函数
        mask = torch.zeros(idx.size(0), idx.size(0)).to(self.device)    #初始化掩码（为什么使用掩码？？GWNet里有用吗？）
        mask.fill_(float('0'))  #填充掩码
        s1,t1 = adj.topk(self.k,1)  #topk方法是在哪里定义的？？？
        mask.scatter_(1,t1,s1.fill_(1)) #和GWNet生成稀疏邻接矩阵的方法不一样（为什么？GWNet使用的SVD）
        adj = adj*mask
        return adj



class graph_directed(nn.Module):
    def __init__(self, nnodes, k, dim, device, alpha=3, static_feat=None):
        """
        有向图构造类(和构造无向图代码的区别？？？)
        :param nnodes:节点数量
        :param k:每个节点的top-k邻居
        :param dim:嵌入维度（？？？）
        :param device:设备
        :param alpha:控制斜率
        :param static_feat:静态特征
        """
        super(graph_directed, self).__init__()
        self.nnodes = nnodes
        if static_feat is not None:
            xd = static_feat.shape[1]
            self.lin1 = nn.Linear(xd, dim)
            self.lin2 = nn.Linear(xd, dim)
        else:
            self.emb1 = nn.Embedding(nnodes, dim)   #节点嵌入（没有静态特征的节点嵌入的方法，什么是静态特征，有什么要求？dim取决于什么？）
            self.emb2 = nn.Embedding(nnodes, dim)   #节点嵌入
            self.lin1 = nn.Linear(dim,dim)
            self.lin2 = nn.Linear(dim,dim)

        self.device = device
        self.k = k
        self.dim = dim
        self.alpha = alpha
        self.static_feat = static_feat

    def forward(self, idx):
        """
        前向传播
        :param idx:节点索引，形状为（num_nodes, ）
        :return :稀疏矩阵，形状为（num_nodes, num_nodes）
        """
        if self.static_feat is None:
            nodevec1 = self.emb1(idx)
            nodevec2 = self.emb2(idx)
        else:
            nodevec1 = self.static_feat[idx,:]
            nodevec2 = nodevec1

        nodevec1 = torch.tanh(self.alpha*self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha*self.lin2(nodevec2))

        a = torch.mm(nodevec1, nodevec2.transpose(1,0))
        adj = F.relu(torch.tanh(self.alpha*a))
        mask = torch.zeros(idx.size(0), idx.size(0)).to(self.device)
        mask.fill_(float('0'))
        s1,t1 = adj.topk(self.k,1)
        mask.scatter_(1,t1,s1.fill_(1))
        adj = adj*mask
        return adj


class LayerNorm(nn.Module):
    __constants__ = ['normalized_shape', 'weight', 'bias', 'eps', 'elementwise_affine']
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        """
        层归一化类
        :param normalized_shape:归一化的形状
        :param eps:数值稳定性参数，用于防止除以0
        :param elementwise_affine:是否使用科学系的缩放和平移参数
        """
        super(LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):#如果normalized_shape为整数，则转换为单元素元组(n,)
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.Tensor(*normalized_shape))
            self.bias = nn.Parameter(torch.Tensor(*normalized_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()


    def reset_parameters(self):
        if self.elementwise_affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def forward(self, input, idx):
        if self.elementwise_affine:
            return F.layer_norm(input, tuple(input.shape[1:]), self.weight[:,idx,:], self.bias[:,idx,:], self.eps)
        else:
            return F.layer_norm(input, tuple(input.shape[1:]), self.weight, self.bias, self.eps)

    def extra_repr(self):
        return '{normalized_shape}, eps={eps}, ' \
            'elementwise_affine={elementwise_affine}'.format(**self.__dict__)
