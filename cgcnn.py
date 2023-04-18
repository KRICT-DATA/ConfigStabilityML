# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 11:19:59 2022

@author: User
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean

class ConvLayer(nn.Module):
    def __init__(self,node_fea_len,edge_fea_len):
        super(ConvLayer,self).__init__()
        self.node_fea_len = node_fea_len
        self.edge_fea_len = edge_fea_len
       
        self.fc_full = nn.Linear(2*self.node_fea_len+self.edge_fea_len,2*self.node_fea_len)
        self.sigmoid = nn.Sigmoid()
        self.f1 = nn.Softplus()
        self.bn1 = nn.BatchNorm1d(2*self.node_fea_len)
        self.bn2 = nn.BatchNorm1d(self.node_fea_len)
        self.f2 = nn.Softplus()
        
    def forward(self,node_fea,idx1,idx2,edge_fea):
        node1 = node_fea[idx1]
        node2 = node_fea[idx2]
        z12 = torch.cat([node1,node2,edge_fea],dim=1)
        
        z12 = self.bn1(self.fc_full(z12))
        gate,conv_fea = z12.chunk(2,dim=1)
        gate = self.sigmoid(gate)
        conv_fea = self.f1(conv_fea)
        
        node_new = self.bn2(scatter_mean(gate*conv_fea,idx1,dim=0,out=torch.zeros_like(node_fea)))
        node_fea = self.f2(node_fea+node_new)
        return node_fea
    
class CGCNN(nn.Module):
    def __init__(self,node_fea_len,edge_fea_len,n_conv,h_fea_len,n_h):
        super(CGCNN,self).__init__()
        self.embedding = nn.Embedding(101,node_fea_len)
        self.convs = nn.ModuleList([ConvLayer(node_fea_len,edge_fea_len) for _ in range(n_conv)])
        
        self.conv_to_fc = nn.Linear(node_fea_len,h_fea_len)
        self.conv_to_fc_f = nn.Softplus()
        
        if n_h > 1:
            self.fcs = nn.ModuleList([nn.Linear(h_fea_len,h_fea_len) for _ in range(n_h-1)])
            self.fs = nn.ModuleList([nn.Softplus() for _ in range(n_h-1)])
        
        self.fc_out = nn.Linear(h_fea_len,2)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self,node_fea,edge_fea,idx1,idx2,idx3):
        node_fea = self.embedding(node_fea)
        for conv_func in self.convs:
            node_fea = conv_func(node_fea,idx1,idx2,edge_fea)
            
        crys_fea = scatter_mean(node_fea,idx3,dim=0)
        crys_fea = self.conv_to_fc_f(self.conv_to_fc(crys_fea))
        
        if hasattr(self,'fcs'):
            for fc,f in zip(self.fcs,self.fs):
                crys_fea = f(fc(crys_fea))
                
        out = self.fc_out(crys_fea)
        return out
    
        
