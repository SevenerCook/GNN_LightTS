from GNN_based.layer import *


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.gcn_true = args.gcn_true
        self.buildA_true = args.buildA_true
        self.gcn_depth = args.gcn_depth
        self.num_nodes = args.num_nodes
        self.dropout = args.dropout
        self.subgraph_size = args.subgraph_size
        self.node_dim = args.node_dim
        self.dilation_exponential = args.dilation_exponential
        self.conv_channels = args.conv_channels
        self.residual_channels = args.residual_channels
        self.skip_channels = args.skip_channels
        self.end_channels = args.end_channels
        self.predefined_A = None
        self.static_feat = None
        self.in_dim = args.in_dim
        self.out_dim = args.pred_len
        self.propalpha = args.propalpha
        self.tanhalpha = args.tanhalpha
        self.layers = 3
        self.layer_norm_affline = True
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.gconv1 = nn.ModuleList()
        self.gconv2 = nn.ModuleList()
        self.norm = nn.ModuleList()
        self.start_conv = nn.Conv2d(in_channels=self.in_dim,
                                    out_channels=self.residual_channels,
                                    kernel_size=(1, 1))
        self.gc = graph_constructor(self.num_nodes, self.subgraph_size, self.node_dim, args.devices, alpha=args.tanhalpha, static_feat=self.static_feat)

        self.seq_length = args.seq_in_len
        kernel_size = 7
        if self.dilation_exponential>1:
            self.receptive_field = int(1+ (kernel_size-1)*(self.dilation_exponential**self.layers-1)/(self.dilation_exponential-1))
        else:
            self.receptive_field = self.layers*(kernel_size-1) + 1

        for i in range(1):
            if self.dilation_exponential>1:
                rf_size_i = int(1 + i*(kernel_size-1)*(self.dilation_exponential**self.layers-1)/(self.dilation_exponential-1))
            else:
                rf_size_i = i*self.layers*(kernel_size-1)+1
            new_dilation = 1
            for j in range(1,self.layers+1):
                #dialated_conv(即TCN卷积)
                if self.dilation_exponential > 1:
                    rf_size_j = int(rf_size_i + (kernel_size-1)*(self.dilation_exponential**j-1)/(self.dilation_exponential-1))
                else:
                    rf_size_j = rf_size_i+j*(kernel_size-1)
                #常规必要的过滤卷积，GCModule门控卷积,残差卷积
                self.filter_convs.append(dilated_inception(self.residual_channels, self.conv_channels, dilation_factor=new_dilation))
                self.gate_convs.append(dilated_inception(self.residual_channels, self.conv_channels, dilation_factor=new_dilation))
                self.residual_convs.append(nn.Conv2d(in_channels=self.conv_channels,
                                                    out_channels=self.residual_channels,
                                                 kernel_size=(1, 1)))
                #添加skip_connection
                if self.seq_length>self.receptive_field:
                    self.skip_convs.append(nn.Conv2d(in_channels=self.conv_channels,
                                                    out_channels=self.skip_channels,
                                                    kernel_size=(1, self.seq_length-rf_size_j+1)))
                else:
                    self.skip_convs.append(nn.Conv2d(in_channels=self.conv_channels,
                                                    out_channels=self.skip_channels,
                                                    kernel_size=(1, self.receptive_field-rf_size_j+1)))
                #是否添加图卷积神经网络
                if self.gcn_true:
                    self.gconv1.append(mixprop(self.conv_channels, self.residual_channels, self.gcn_depth, self.dropout, self.propalpha))
                    self.gconv2.append(mixprop(self.conv_channels, self.residual_channels, self.gcn_depth, self.dropout, self.propalpha))
                #归一化层
                if self.seq_length>self.receptive_field:
                    self.norm.append(LayerNorm((self.residual_channels, self.num_nodes, self.seq_length - rf_size_j + 1),elementwise_affine=self.layer_norm_affline))
                else:
                    self.norm.append(LayerNorm((self.residual_channels, self.num_nodes, self.receptive_field - rf_size_j + 1),elementwise_affine=self.layer_norm_affline))

                new_dilation *= self.dilation_exponential

        self.layers = 3   #控制模块组的数量
        #跳跃连接信息的卷积操作
        self.end_conv_1 = nn.Conv2d(in_channels=self.skip_channels,
                                             out_channels=self.end_channels,
                                             kernel_size=(1,1),
                                             bias=True)
        #最后一层
        self.end_conv_2 = nn.Conv2d(in_channels=self.end_channels,
                                             out_channels=self.out_dim,
                                             kernel_size=(1,1),
                                             bias=True) #out_channels=self.out_dim
        #？？？？
        if self.seq_length > self.receptive_field:
            #源输入信息的跳跃连接卷积处理
            self.skip0 = nn.Conv2d(in_channels=self.in_dim, out_channels=self.skip_channels, kernel_size=(1, self.seq_length), bias=True)
            #残差信息的跳跃连接卷积处理
            self.skipE = nn.Conv2d(in_channels=self.residual_channels, out_channels=self.skip_channels, kernel_size=(1, self.seq_length-self.receptive_field+1), bias=True)

        else:
            self.skip0 = nn.Conv2d(in_channels=self.in_dim, out_channels=self.skip_channels, kernel_size=(1, self.receptive_field), bias=True)
            self.skipE = nn.Conv2d(in_channels=self.residual_channels, out_channels=self.skip_channels, kernel_size=(1, 1), bias=True)


        self.idx = torch.arange(self.num_nodes).to(0)#没有邻接矩阵，手动创建出用于生成邻接矩阵


    def forward(self, input, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None, idx=None):
        """
        前向传播
        :param input:源输入信息，形状（batch_size, num_channels, num_features, seq_length）(32,1,7,12)
        :param idx:是否指定特定的节点
        return x:（batch_size，seq_length， num_features, num_channels）（32，12，7，1）
        """
        input = input.transpose(1, 2)
        input = input.unsqueeze(1)
        seq_len = input.size(3)
        assert seq_len==self.seq_length, 'input sequence length not equal to preset sequence length'

        #填充，以便于空洞卷积的使用（self.receptive_field是设定的？）
        if self.seq_length<self.receptive_field:
            input = nn.functional.pad(input,(self.receptive_field-self.seq_length,0,0,0))


        #首先构建邻接矩阵
        if self.gcn_true:
            if self.buildA_true:
                if idx is None:
                    adp = self.gc(self.idx)
                else:
                    adp = self.gc(idx)
            else:
                adp = self.predefined_A
        #初始化卷积层
        x = self.start_conv(input)                                                  #input->x channel（in_dim(2 -> res_dim(32）
        #源输入的跳跃连接层
        skip = self.skip0(F.dropout(input, self.dropout, training=self.training))   #input->skip channel(in_dim(2 -> skip_channel(32 )

        for i in range(self.layers):
            residual = x
            filter = self.filter_convs[i](x)                                        #x->filter channel(res_chan(32 -> conv_chan(32 )
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](x)                                            #x->gate channel(res_chan(32 -> conv_chan(32 )
            gate = torch.sigmoid(gate)
            x = filter * gate
            x = F.dropout(x, self.dropout, training=self.training)
            s = x
            s = self.skip_convs[i](s)                                               #x==s->s channel(in_dim(2 -> skip_channel(32 )
            skip = s + skip
            if self.gcn_true:
                x = self.gconv1[i](x, adp)+self.gconv2[i](x, adp.transpose(1,0))    #x->x     channel(conv_chan(32 -> res_chan(32)
            else:
                x = self.residual_convs[i](x)

            x = x + residual[:, :, :, -x.size(3):]                                  #x = x+residual
            if idx is None:
                x = self.norm[i](x,self.idx)                                        #layernorm x->x (res_chan(32 -> num_nodes(7 )起到了维度的什么作用？怎么改变第三维度大小的？
            else:
                x = self.norm[i](x,idx)

        skip = self.skipE(x) + skip
        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))                                              #x->x (skip_chan(32 ->end_chan(128)
        x = self.end_conv_2(x)

        #x->x (end_chan(32 ->out_dim(12 )(32,12,7,1)
        x = x.view(x.size(0), x.size(1), -1)

        return x




class Model2(nn.Module):
    def __init__(self, args):
        super(Model2, self).__init__()
        self.gcn_true = args.gcn_true
        self.buildA_true = args.buildA_true
        self.gcn_depth = args.gcn_depth
        self.num_nodes = args.num_nodes
        self.dropout = args.dropout
        self.subgraph_size = args.subgraph_size
        self.node_dim = args.node_dim
        self.dilation_exponential = args.dilation_exponential
        self.conv_channels = args.conv_channels
        self.residual_channels = args.residual_channels
        self.skip_channels = args.skip_channels
        self.end_channels = args.end_channels
        self.predefined_A = None
        self.static_feat = None
        self.in_dim = args.in_dim
        self.out_dim = args.pred_len
        self.propalpha = args.propalpha
        self.tanhalpha = args.tanhalpha
        self.layers = 3
        self.layer_norm_affline = True
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.gconv1 = nn.ModuleList()
        self.gconv2 = nn.ModuleList()
        self.norm = nn.ModuleList()
        self.start_conv = nn.Conv2d(in_channels=self.in_dim,
                                    out_channels=self.residual_channels,
                                    kernel_size=(1, 1))
        self.gc = graph_constructor(self.num_nodes, self.subgraph_size, self.node_dim, args.devices, alpha=args.tanhalpha, static_feat=self.static_feat)

        self.seq_length = args.seq_in_len
        kernel_size = 7
        if self.dilation_exponential>1:
            self.receptive_field = int(1+ (kernel_size-1)*(self.dilation_exponential**self.layers-1)/(self.dilation_exponential-1))
        else:
            self.receptive_field = self.layers*(kernel_size-1) + 1

        for i in range(1):
            if self.dilation_exponential>1:
                rf_size_i = int(1 + i*(kernel_size-1)*(self.dilation_exponential**self.layers-1)/(self.dilation_exponential-1))
            else:
                rf_size_i = i*self.layers*(kernel_size-1)+1
            new_dilation = 1
            for j in range(1,self.layers+1):
                #dialated_conv(即TCN卷积)
                if self.dilation_exponential > 1:
                    rf_size_j = int(rf_size_i + (kernel_size-1)*(self.dilation_exponential**j-1)/(self.dilation_exponential-1))
                else:
                    rf_size_j = rf_size_i+j*(kernel_size-1)
                #常规必要的过滤卷积，GCModule门控卷积,残差卷积
                self.filter_convs.append(dilated_inception(self.residual_channels, self.conv_channels, dilation_factor=new_dilation))
                self.gate_convs.append(dilated_inception(self.residual_channels, self.conv_channels, dilation_factor=new_dilation))
                self.residual_convs.append(nn.Conv2d(in_channels=self.conv_channels,
                                                    out_channels=self.residual_channels,
                                                 kernel_size=(1, 1)))
                #添加skip_connection
                if self.seq_length>self.receptive_field:
                    self.skip_convs.append(nn.Conv2d(in_channels=self.conv_channels,
                                                    out_channels=self.skip_channels,
                                                    kernel_size=(1, self.seq_length-rf_size_j+1)))
                else:
                    self.skip_convs.append(nn.Conv2d(in_channels=self.conv_channels,
                                                    out_channels=self.skip_channels,
                                                    kernel_size=(1, self.receptive_field-rf_size_j+1)))
                #是否添加图卷积神经网络
                if self.gcn_true:
                    self.gconv1.append(mixprop(self.conv_channels, self.residual_channels, self.gcn_depth, self.dropout, self.propalpha))
                    self.gconv2.append(mixprop(self.conv_channels, self.residual_channels, self.gcn_depth, self.dropout, self.propalpha))
                #归一化层
                if self.seq_length>self.receptive_field:
                    self.norm.append(LayerNorm((self.residual_channels, self.num_nodes, self.seq_length - rf_size_j + 1),elementwise_affine=self.layer_norm_affline))
                else:
                    self.norm.append(LayerNorm((self.residual_channels, self.num_nodes, self.receptive_field - rf_size_j + 1),elementwise_affine=self.layer_norm_affline))

                new_dilation *= self.dilation_exponential

        self.layers = 3   #控制模块组的数量
        #跳跃连接信息的卷积操作
        self.end_conv_1 = nn.Conv2d(in_channels=self.skip_channels,
                                             out_channels=self.end_channels,
                                             kernel_size=(1,1),
                                             bias=True)
        #最后一层
        self.end_conv_2 = nn.Conv2d(in_channels=self.end_channels,
                                             out_channels=96,
                                             kernel_size=(1,1),
                                             bias=True) #out_channels=self.out_dim
        #？？？？
        if self.seq_length > self.receptive_field:
            #源输入信息的跳跃连接卷积处理
            self.skip0 = nn.Conv2d(in_channels=self.in_dim, out_channels=self.skip_channels, kernel_size=(1, self.seq_length), bias=True)
            #残差信息的跳跃连接卷积处理
            self.skipE = nn.Conv2d(in_channels=self.residual_channels, out_channels=self.skip_channels, kernel_size=(1, self.seq_length-self.receptive_field+1), bias=True)

        else:
            self.skip0 = nn.Conv2d(in_channels=self.in_dim, out_channels=self.skip_channels, kernel_size=(1, self.receptive_field), bias=True)
            self.skipE = nn.Conv2d(in_channels=self.residual_channels, out_channels=self.skip_channels, kernel_size=(1, 1), bias=True)


        self.idx = torch.arange(self.num_nodes).to(0)#没有邻接矩阵，手动创建出用于生成邻接矩阵


    def forward(self, input, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None, idx=None):
        """
        前向传播
        :param input:源输入信息，形状（batch_size, num_channels, num_features, seq_length）(32,1,7,12)
        :param idx:是否指定特定的节点
        return x:（batch_size，seq_length， num_features, num_channels）（32，12，7，1）
        """
        input = input.transpose(1, 2)
        input = input.unsqueeze(1)
        seq_len = input.size(3)
        assert seq_len==self.seq_length, 'input sequence length not equal to preset sequence length'

        #填充，以便于空洞卷积的使用（self.receptive_field是设定的？）
        if self.seq_length<self.receptive_field:
            input = nn.functional.pad(input,(self.receptive_field-self.seq_length,0,0,0))


        #首先构建邻接矩阵
        if self.gcn_true:
            if self.buildA_true:
                if idx is None:
                    adp = self.gc(self.idx)
                else:
                    adp = self.gc(idx)
            else:
                adp = self.predefined_A
        #初始化卷积层
        x = self.start_conv(input)                                                  #input->x channel（in_dim(2 -> res_dim(32）
        #源输入的跳跃连接层
        skip = self.skip0(F.dropout(input, self.dropout, training=self.training))   #input->skip channel(in_dim(2 -> skip_channel(32 )

        for i in range(self.layers):
            residual = x
            filter = self.filter_convs[i](x)                                        #x->filter channel(res_chan(32 -> conv_chan(32 )
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](x)                                            #x->gate channel(res_chan(32 -> conv_chan(32 )
            gate = torch.sigmoid(gate)
            x = filter * gate
            x = F.dropout(x, self.dropout, training=self.training)
            s = x
            s = self.skip_convs[i](s)                                               #x==s->s channel(in_dim(2 -> skip_channel(32 )
            skip = s + skip
            if self.gcn_true:
                x = self.gconv1[i](x, adp)+self.gconv2[i](x, adp.transpose(1,0))    #x->x     channel(conv_chan(32 -> res_chan(32)
            else:
                x = self.residual_convs[i](x)

            x = x + residual[:, :, :, -x.size(3):]                                  #x = x+residual
            if idx is None:
                x = self.norm[i](x,self.idx)                                        #layernorm x->x (res_chan(32 -> num_nodes(7 )起到了维度的什么作用？怎么改变第三维度大小的？
            else:
                x = self.norm[i](x,idx)

        skip = self.skipE(x) + skip
        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))                                              #x->x (skip_chan(32 ->end_chan(128)
        x = self.end_conv_2(x)

        #x->x (end_chan(32 ->out_dim(12 )(32,12,7,1)
        x = x.view(x.size(0), x.size(1), -1)

        return x