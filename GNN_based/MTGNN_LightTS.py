import torch
import pandas as pd
import numpy as np
from openpyxl.styles.builtins import output
from torch import nn
from GNN_based import net,CNN
#from GNN_based.GWNet import Model
from models import Autoformer_1,simple_linear,iTransformer,LightTS


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.graphLayer = net.Model2(args)
        self.MLPlayer = LightTS.Model(args)
        self.feature_extracter = CNN.Basic1DCNN

    def forward(self,input, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None, idx=None):
        """

        """
        output1 = self.graphLayer(input, x_mark_enc, x_dec, x_mark_dec)
        output2 = self.MLPlayer(output1, x_mark_enc, x_dec, x_mark_dec)
        return output2
