import torch
import torch.nn as nn
import torch.nn.functional as F
import math

#自定义二维卷积层，支持SAME填充和批量归一化——数据尺寸不变的卷积和批量归一化方式（适用于作为第一步数据处理）
class conv2d_(nn.Module):
    def __init__(self, input_dims, output_dims, kernel_size, stride=(1, 1),
                 padding='SAME', use_bias=True, activation=F.relu,
                 bn_decay=None):
        super(conv2d_, self).__init__()
        self.activation = activation
        # 计算SAME模式下的填充尺寸
        if padding == 'SAME':
            self.padding_size = math.ceil(kernel_size)
        else:
            self.padding_size = [0, 0]
        self.conv = nn.Conv2d(input_dims, output_dims, kernel_size, stride=stride,
                              padding=0, bias=use_bias)
        self.batch_norm = nn.BatchNorm2d(output_dims, momentum=bn_decay)
        torch.nn.init.xavier_uniform_(self.conv.weight)

        if use_bias:
            torch.nn.init.zeros_(self.conv.bias)


    def forward(self, x):
        """
        前向传播
        :param x:[B,H,W,C]
        """
        x = x.permute(0, 3, 2, 1)   #[B,C,H,W]
        #对称填充（上下左右）
        x = F.pad(x, ([self.padding_size[1], self.padding_size[1], self.padding_size[0], self.padding_size[0]]))
        x = self.conv(x)
        x = self.batch_norm(x)
        if self.activation is not None:
            x = F.relu_(x)
        return x.permute(0, 3, 2, 1)    #恢复维度[B,H,W,C]

# 全连接层（1x1卷积实现），快速建立n层全连接层
class FC(nn.Module):
    def __init__(self, input_dims, units, activations, bn_decay, use_bias=True):
        super(FC, self).__init__()
        if isinstance(units, int):
            units = [units]
            input_dims = [input_dims]
            activations = [activations]
        elif isinstance(units, tuple):
            units = list(units)
            input_dims = list(input_dims)
            activations = list(activations)
        assert type(units) == list
        self.convs = nn.ModuleList([conv2d_(
            input_dims=input_dim, output_dims=num_unit, kernel_size=[1, 1], stride=[1, 1],
            padding='VALID', use_bias=use_bias, activation=activation,
            bn_decay=bn_decay) for input_dim, num_unit, activation in
            zip(input_dims, units, activations)])

    def forward(self, x):
        for conv in self.convs:
            x = conv(x)
        return x

#时空嵌入模块（融合空间和时间特征）——嵌入原理
class STEmbedding(nn.Module):
    '''
    spatio-temporal embedding
    SE:     [num_vertex, D]
    TE:     [batch_size, num_his + num_pred, 2] (dayofweek, timeofday)
    T:      num of time steps in one day
    D:      output dims
    retrun: [batch_size, num_his + num_pred, num_vertex, D]
    '''

    def __init__(self, D, bn_decay):
        super(STEmbedding, self).__init__()
        # 空间嵌入的全连接层
        self.FC_se = FC(
            input_dims=[D, D], units=[D, D], activations=[F.relu, None],
            bn_decay=bn_decay)
        # 时间嵌入的全连接层
        self.FC_te = FC(
            input_dims=[295, D], units=[D, D], activations=[F.relu, None],
            bn_decay=bn_decay)  # input_dims = time step per day + days per week=288+7=295

    def forward(self, SE, TE, T=288):
        """
        前向传播
        288=24*12(为什么这样分？)
        :param SE: [num_vertex, D]
        :param TE: [batch_size, num_steps, 2](两列分别表示星期几和时刻)
        return :[batch_size, num_steps, num_vertex, D]
        """
        # spatial embedding(添加批次和时间维度)
        SE = SE.unsqueeze(0).unsqueeze(0)
        SE = self.FC_se(SE) # 保持维度不变

        # temporal embedding（时间特征转换为one-hot）
        dayofweek = torch.empty(TE.shape[0], TE.shape[1], 7)
        timeofday = torch.empty(TE.shape[0], TE.shape[1], T)
        for i in range(TE.shape[0]):
            dayofweek[i] = F.one_hot(TE[..., 0][i].to(torch.int64) % 7, 7)
        for j in range(TE.shape[0]):
            timeofday[j] = F.one_hot(TE[..., 1][j].to(torch.int64) % 288, T)
        # 拼接时间特征，拼接在同一维度
        TE = torch.cat((dayofweek, timeofday), dim=-1)
        TE = TE.unsqueeze(dim=2)
        TE = self.FC_te(TE)
        del dayofweek, timeofday
        return SE + TE

# 空间注意力机制，存在自注意力机制
class spatialAttention(nn.Module):
    '''
    spatial attention mechanism
    X:      [batch_size, num_step, num_vertex, D]
    STE:    [batch_size, num_step, num_vertex, D]
    K:      number of attention heads
    d:      dimension of each attention outputs
    return: [batch_size, num_step, num_vertex, D]
    '''

    def __init__(self, K, d, bn_decay):
        super(spatialAttention, self).__init__()
        D = K * d
        self.d = d  #单头维度
        self.K = K  #多头数量
        # 定义Q,K,V的全连接层
        self.FC_q = FC(input_dims=2 * D, units=D, activations=F.relu,
                       bn_decay=bn_decay)
        self.FC_k = FC(input_dims=2 * D, units=D, activations=F.relu,
                       bn_decay=bn_decay)
        self.FC_v = FC(input_dims=2 * D, units=D, activations=F.relu,
                       bn_decay=bn_decay)
        self.FC = FC(input_dims=D, units=D, activations=F.relu,
                     bn_decay=bn_decay)

    def forward(self, X, STE):
        """
        前向传播
        :param X: [batch_size, num_steps, num_vertex, D]
        :param STE:时空嵌入 [batch_size, num_steps, num_vertex, D]
        return :[batch_size, num_steps, num_vertex, D]
        """
        batch_size = X.shape[0]
        X = torch.cat((X, STE), dim=-1)
        # [batch_size, num_steps, num_vertex, 2D]=》[batch_size, num_step, num_vertex, K * d]
        query = self.FC_q(X)
        key = self.FC_k(X)
        value = self.FC_v(X)
        # [batch_size, num_step, num_vertex, K * d]=》[K * batch_size, num_step, num_vertex, d]
        query = torch.cat(torch.split(query, self.K, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.K, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.K, dim=-1), dim=0)
        # [K * batch_size, num_step, num_vertex, num_vertex]
        attention = torch.matmul(query, key.transpose(2, 3))
        attention /= (self.d ** 0.5)
        attention = F.softmax(attention, dim=-1)
        # [batch_size, num_step, num_vertex, D]
        X = torch.matmul(attention, value)
        X = torch.cat(torch.split(X, batch_size, dim=0), dim=-1)  # orginal K, change to batch_size
        X = self.FC(X)
        del query, key, value, attention
        return X

# 时间注意力机制（作用在时间维度），交叉注意力机制
class temporalAttention(nn.Module):
    '''
    temporal attention mechanism
    X:      [batch_size, num_step, num_vertex, D]
    STE:    [batch_size, num_step, num_vertex, D]
    K:      number of attention heads
    d:      dimension of each attention outputs
    return: [batch_size, num_step, num_vertex, D]
    '''

    def __init__(self, K, d, bn_decay, mask=True):
        super(temporalAttention, self).__init__()
        D = K * d
        self.d = d
        self.K = K
        self.mask = mask
        self.FC_q = FC(input_dims=2 * D, units=D, activations=F.relu,
                       bn_decay=bn_decay)
        self.FC_k = FC(input_dims=2 * D, units=D, activations=F.relu,
                       bn_decay=bn_decay)
        self.FC_v = FC(input_dims=2 * D, units=D, activations=F.relu,
                       bn_decay=bn_decay)
        self.FC = FC(input_dims=D, units=D, activations=F.relu,
                     bn_decay=bn_decay)

    def forward(self, X, STE):
        batch_size_ = X.shape[0]
        X = torch.cat((X, STE), dim=-1)
        # [batch_size, num_step, num_vertex, K * d]
        query = self.FC_q(X)
        key = self.FC_k(X)
        value = self.FC_v(X)
        # [K * batch_size, num_step, num_vertex, d]
        query = torch.cat(torch.split(query, self.K, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.K, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.K, dim=-1), dim=0)
        # query: [K * batch_size, num_vertex, num_step, d]
        # key:   [K * batch_size, num_vertex, d, num_step]
        # value: [K * batch_size, num_vertex, num_step, d]
        query = query.permute(0, 2, 1, 3)
        key = key.permute(0, 2, 3, 1)
        value = value.permute(0, 2, 1, 3)
        # [K * batch_size, num_vertex, num_step, num_step]
        attention = torch.matmul(query, key)
        attention /= (self.d ** 0.5)
        # mask attention score
        if self.mask:
            batch_size = X.shape[0]
            num_step = X.shape[1]
            num_vertex = X.shape[2]
            mask = torch.ones(num_step, num_step)
            mask = torch.tril(mask)
            mask = torch.unsqueeze(torch.unsqueeze(mask, dim=0), dim=0)
            mask = mask.repeat(self.K * batch_size, num_vertex, 1, 1)
            mask = mask.to(torch.bool)
            attention = torch.where(mask, attention, -2 ** 15 + 1)
        # softmax
        attention = F.softmax(attention, dim=-1)
        # [batch_size, num_step, num_vertex, D]
        X = torch.matmul(attention, value)
        X = X.permute(0, 2, 1, 3)
        X = torch.cat(torch.split(X, batch_size_, dim=0), dim=-1)  # orginal K, change to batch_size
        X = self.FC(X)
        del query, key, value, attention
        return X

#门控融合模块（结合空间和时间注意力结果），自注意力机制的结果
class gatedFusion(nn.Module):
    '''
    gated fusion
    HS:     [batch_size, num_step, num_vertex, D]
    HT:     [batch_size, num_step, num_vertex, D]
    D:      output dims
    return: [batch_size, num_step, num_vertex, D]
    '''

    def __init__(self, D, bn_decay):
        super(gatedFusion, self).__init__()
        self.FC_xs = FC(input_dims=D, units=D, activations=None,
                        bn_decay=bn_decay, use_bias=False)
        self.FC_xt = FC(input_dims=D, units=D, activations=None,
                        bn_decay=bn_decay, use_bias=True)
        self.FC_h = FC(input_dims=[D, D], units=[D, D], activations=[F.relu, None],
                       bn_decay=bn_decay)

    def forward(self, HS, HT):
        """
        前向传播
        :param HS:空间注意力输出 [batch_size, num_step, num_vertex, D]
        :param HT:时间注意力输出 [batch_size, num_step, num_vertex, D]
        :return H:[batch_size, num_step, num_vertex, D]
        """
        XS = self.FC_xs(HS)
        XT = self.FC_xt(HT)
        z = torch.sigmoid(torch.add(XS, XT))
        H = torch.add(torch.mul(z, HS), torch.mul(1 - z, HT))
        H = self.FC_h(H)
        del XS, XT, z
        return H

# 时空注意力块（残差连接）为什么要进行残差连接，想要利用上一层的信息
class STAttBlock(nn.Module):
    def __init__(self, K, d, bn_decay, mask=False):
        super(STAttBlock, self).__init__()
        self.spatialAttention = spatialAttention(K, d, bn_decay)
        self.temporalAttention = temporalAttention(K, d, bn_decay, mask=mask)
        self.gatedFusion = gatedFusion(K * d, bn_decay)

    def forward(self, X, STE):
        HS = self.spatialAttention(X, STE)
        HT = self.temporalAttention(X, STE)
        H = self.gatedFusion(HS, HT)
        del HS, HT
        # 进行残差连接
        return torch.add(X, H)

# 变换注意力（连接编码器和解码器），就是解码器
class transformAttention(nn.Module):
    '''
    transform attention mechanism
    X:        [batch_size, num_his, num_vertex, D]
    STE_his:  [batch_size, num_his, num_vertex, D]
    STE_pred: [batch_size, num_pred, num_vertex, D]
    K:        number of attention heads
    d:        dimension of each attention outputs
    return:   [batch_size, num_pred, num_vertex, D]
    '''

    def __init__(self, K, d, bn_decay):
        super(transformAttention, self).__init__()
        D = K * d
        self.K = K
        self.d = d
        self.FC_q = FC(input_dims=D, units=D, activations=F.relu,
                       bn_decay=bn_decay)
        self.FC_k = FC(input_dims=D, units=D, activations=F.relu,
                       bn_decay=bn_decay)
        self.FC_v = FC(input_dims=D, units=D, activations=F.relu,
                       bn_decay=bn_decay)
        self.FC = FC(input_dims=D, units=D, activations=F.relu,
                     bn_decay=bn_decay)

    def forward(self, X, STE_his, STE_pred):
        """
        前向传播
        :param X:编码器输出    [batch_size, num_his, num_vertex, D]
        :param STE_his:历史时间嵌入 [batch_size, num_his, num_vertex, D]
        :param STE_pred:预测时间嵌入 [batch_size, num_pred, num_vertex, D]
        return: [batch_size, num_pred, num_vertex, D]
        """
        batch_size = X.shape[0]
        # [batch_size, num_step, num_vertex, K * d]
        query = self.FC_q(STE_pred)
        key = self.FC_k(STE_his)
        value = self.FC_v(X)
        # [K * batch_size, num_step, num_vertex, d]
        query = torch.cat(torch.split(query, self.K, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.K, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.K, dim=-1), dim=0)
        # query: [K * batch_size, num_vertex, num_pred, d]
        # key:   [K * batch_size, num_vertex, d, num_his]
        # value: [K * batch_size, num_vertex, num_his, d]
        query = query.permute(0, 2, 1, 3)
        key = key.permute(0, 2, 3, 1)
        value = value.permute(0, 2, 1, 3)
        # [K * batch_size, num_vertex, num_pred, num_his]
        attention = torch.matmul(query, key)
        attention /= (self.d ** 0.5)
        attention = F.softmax(attention, dim=-1)
        # [batch_size, num_pred, num_vertex, D]
        X = torch.matmul(attention, value)
        X = X.permute(0, 2, 1, 3)
        X = torch.cat(torch.split(X, batch_size, dim=0), dim=-1)
        X = self.FC(X)
        del query, key, value, attention
        return X


class GMAN(nn.Module):
    '''
    GMAN
        X：       [batch_size, num_his, num_vertx]
        TE：      [batch_size, num_his + num_pred, 2] (time-of-day, day-of-week)
        SE：      [num_vertex, K * d]
        num_his： number of history steps
        num_pred：number of prediction steps
        T：       one day is divided into T steps
        L：       number of STAtt blocks in the encoder/decoder
        K：       number of attention heads
        d：       dimension of each attention head outputs
        return：  [batch_size, num_pred, num_vertex]
    '''
    def __init__(self, SE, args, bn_decay):
        super(GMAN, self).__init__()
        L = args.L  #STAtt块层数
        K = args.K  #注意力头数
        d = args.d  #单头维度
        D = K * d   #总特征维度
        self.num_his = args.num_his #历史时间步数
        # 模型组件
        self.SE = SE    # 空间嵌入矩阵（num_vertex,D）
        self.STEmbedding = STEmbedding(D, bn_decay)
        self.STAttBlock_1 = nn.ModuleList([STAttBlock(K, d, bn_decay) for _ in range(L)])
        self.STAttBlock_2 = nn.ModuleList([STAttBlock(K, d, bn_decay) for _ in range(L)])
        self.transformAttention = transformAttention(K, d, bn_decay)
        # 输入输出变换层
        self.FC_1 = FC(input_dims=[1, D], units=[D, D], activations=[F.relu, None],
                       bn_decay=bn_decay)
        self.FC_2 = FC(input_dims=[D, D], units=[D, 1], activations=[F.relu, None],
                       bn_decay=bn_decay)

    def forward(self, X, TE):
        """
        前向传播
        :param X:输入[batch, num_his, num_vertex]
        :param TE: 时间嵌入[batch, num_his + num_pred, 2]
        return: [batch, num_pred, num_vertex]
        """
        # input 输入变换
        X = torch.unsqueeze(X, -1)  #[batch, num_his, num_vertex, 1]
        X = self.FC_1(X)            #[batch, num_his, num_vertex, D]
        # STE
        STE = self.STEmbedding(self.SE, TE) # [batch, total_steps, num_vertex, D]
        STE_his = STE[:, :self.num_his]     # 历史部分
        STE_pred = STE[:, self.num_his:]    # 预测部分
        # encoder
        for net in self.STAttBlock_1:
            X = net(X, STE_his) # 逐步更新特征，做n次相关性
        # transAtt 时间维度转换
        X = self.transformAttention(X, STE_his, STE_pred) #queries:STE_pred, keys:STE_his, values:X => cross_attention， 返回预测值
        # decoder 解码器处理
        for net in self.STAttBlock_2:
            X = net(X, STE_pred)                         #将时间特征嵌入到对应的预测值上进行自注意力机制，增强了时间特征
        # output 输出变换
        X = self.FC_2(X)    # [batch, num_pred, num_vertex, 1]
        del STE, STE_his, STE_pred
        return torch.squeeze(X, 3)

#GMAN类中的SE如何处理？
#输入的TE就是时间，在这里进行embedding
#这里没有自注意力机制，仅有时空注意力机制
#X是历史数据的特征，为什么这里没有而Autoformer有？
#为什么要进行掩蔽多头注意力机制？1.进行自己特征提取，方便与自注意力机制的提取特征相对应（GMAN中的并没有进行STE_his和STE_pred的自相关，将时间特征嵌入到了X中进行了history自相关）？Autoformer中是怎么处理的？
#       GMAN使用需要预测的未来值的下标进行预测，而Autoformer已经将时间特征融合到特征值中作为一体，使用历史和未来的中间值预测未来值（但他们都有使用在解码器中增强时间特征的结构）