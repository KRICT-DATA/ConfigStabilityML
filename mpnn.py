# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 17:12:51 2022

@author: User
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean

class ConvLayer(nn.Module):
    def __init__(self,dim_v,dim_e):
        super(ConvLayer,self).__init__()
        self.dim_v = dim_v
        self.dim_e = dim_e
        
        self.bn1 = nn.BatchNorm1d(self.dim_e)
        self.bn2 = nn.BatchNorm1d(self.dim_v)
        
        dim_t = 2*self.dim_v+self.dim_e
        self.phi_e = nn.Sequential(nn.Linear(dim_t,self.dim_e),
                                   nn.LeakyReLU(0.2),
                                   nn.Linear(self.dim_e,self.dim_e),
                                   nn.LeakyReLU(0.2),
                                   nn.Linear(self.dim_e,self.dim_e))
        
        dim_t = self.dim_v+self.dim_e
        self.phi_v = nn.Sequential(nn.Linear(dim_t,self.dim_v),
                                   nn.LeakyReLU(0.2),
                                   nn.Linear(self.dim_v,self.dim_v),
                                   nn.LeakyReLU(0.2),
                                   nn.Linear(self.dim_v,self.dim_v))
        
    def forward(self,node_fea,idx1,idx2,edge_fea):
        node1 = node_fea[idx1]
        node2 = node_fea[idx2]
        
        #edge update
        z = torch.cat([node1,node2,edge_fea],dim=1)
        ek_prime = self.bn1(self.phi_e(z))
        edge_new = edge_fea + ek_prime
        
        #node update
        vi_e_bar = scatter_mean(ek_prime,idx1,dim=0,out=torch.zeros_like(node_fea))
        z = torch.cat([vi_e_bar,node_fea],dim=1)
        vi_prime = self.bn2(self.phi_v(z))
        node_new = node_fea + vi_prime
        return node_new,edge_new
    
class MPNN(nn.Module):
    def __init__(self,dim_v0,dim_v1,dim_e0,dim_e1,n_conv,dim_h,n_h):
        super(MPNN,self).__init__()
        self.v_emb = nn.Embedding(dim_v0,dim_v1)
        self.e_emb = nn.Linear(dim_e0,dim_e1)
        
        self.convs = nn.ModuleList([ConvLayer(dim_v1,dim_e1) for _ in range(n_conv)])
        
        dim_t = dim_v1+dim_e1
        self.conv_to_fc = nn.Linear(dim_t,dim_h)
        self.conv_to_fc_f = nn.LeakyReLU(0.2)
        
        if n_h > 1:
            self.fcs = nn.ModuleList([nn.Linear(dim_h,dim_h) for _ in range(n_h-1)])
            self.fs = nn.ModuleList([nn.LeakyReLU(0.2) for _ in range(n_h-1)])
        
        self.fc_out = nn.Linear(dim_h,2)
        
    def forward(self,node_fea,edge_fea,idx1,idx2,idx3):
        node_fea = self.v_emb(node_fea)
        edge_fea = self.e_emb(edge_fea)
        
        for conv_func in self.convs:
            node_fea,edge_fea = conv_func(node_fea,idx1,idx2,edge_fea)
            
        vi_e_bar = scatter_mean(edge_fea,idx1,dim=0,out=torch.zeros_like(node_fea))
            
        crys_fea = torch.cat([vi_e_bar,node_fea],dim=1)
        crys_fea = scatter_mean(crys_fea,idx3,dim=0)
        crys_fea = self.conv_to_fc_f(self.conv_to_fc(crys_fea))
        
        if hasattr(self,'fcs'):
            for fc,f in zip(self.fcs,self.fs):
                crys_fea = f(fc(crys_fea))
                
        out = self.fc_out(crys_fea)
        return out
