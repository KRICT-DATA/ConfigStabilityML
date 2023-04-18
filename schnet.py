# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 11:19:59 2022

@author: User
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean,scatter_sum

class FilterGenerator(nn.Module):
    def __init__(self,dim_e0,dim_e1):
        super(FilterGenerator,self).__init__()
        self.shift = torch.log(torch.tensor(2.0)).item()
        
        self.dense1 = nn.Linear(dim_e0,dim_e1)
        self.dense2 = nn.Linear(dim_e1,dim_e1)

    def forward(self,edge_fea):
        W = self.dense1(edge_fea)
        W = F.softplus(W)-self.shift
        W = self.dense2(W)
        W = F.softplus(W)-self.shift
        return W

class InteractionBlock(nn.Module):
    def __init__(self,dim_v1,dim_e0,dim_e1):
        super(InteractionBlock,self).__init__()
        self.shift = torch.log(torch.tensor(2.0)).item()
        
        self.AtomWise1 = nn.Linear(dim_v1,dim_v1,bias=False)
        self.FilterGen = FilterGenerator(dim_e0,dim_e1)
        self.AtomWise2 = nn.Linear(dim_v1,dim_v1)
        self.AtomWise3 = nn.Linear(dim_v1,dim_v1)
        
    def forward(self,node_fea,idx1,idx2,edge_fea):
        x = self.AtomWise1(node_fea)
        W = self.FilterGen(edge_fea)
        
        cfconf = x[idx2]*W
        node_new = scatter_mean(cfconf,idx1,dim=0,out=torch.zeros_like(node_fea))
        node_new = self.AtomWise2(node_new)
        node_new = F.softplus(node_new)-self.shift
        node_new = self.AtomWise3(node_new)
        
        node_fea = node_fea + node_new
        return node_fea
    
class SchNet(nn.Module):
    def __init__(self,dim_v0,dim_v1,dim_e0,dim_e1,n_conv,h_fea_len,n_h):
        super(SchNet,self).__init__()
        self.shift = torch.log(torch.tensor(2.0)).item()
        
        self.embedding = nn.Embedding(dim_v0,dim_v1)
        self.convs = nn.ModuleList([InteractionBlock(dim_v1,dim_e0,dim_e1) for _ in range(n_conv)])
        
        self.conv_to_fc = nn.Linear(dim_v1,h_fea_len)

        if n_h > 1:
            self.fcs = nn.ModuleList([nn.Linear(h_fea_len,h_fea_len) for _ in range(n_h-1)])
        
        self.fc_out = nn.Linear(h_fea_len,2)
        
    def forward(self,node_fea,edge_fea,idx1,idx2,idx3):
        node_fea = self.embedding(node_fea)
        for conv_func in self.convs:
            node_fea = conv_func(node_fea,idx1,idx2,edge_fea)
            
        crys_fea = scatter_mean(node_fea,idx3,dim=0)
        crys_fea = self.conv_to_fc(crys_fea)
        crys_fea = F.softplus(crys_fea)-self.shift
        
        if hasattr(self,'fcs'):
            for fc in self.fcs:
                crys_fea = fc(crys_fea)
                crys_fea = F.softplus(crys_fea)-self.shift
                
        out = self.fc_out(crys_fea)
        return out
        
