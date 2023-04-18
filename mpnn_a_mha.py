# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 17:12:51 2022

@author: User
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_sum,scatter_mean
from torch_scatter.composite import scatter_softmax

class AttentionHead(nn.Module):
    def __init__(self,dim_v,dim_e,n_head):
        super(AttentionHead,self).__init__()
        self.dim_v = dim_v
        self.dim_e = dim_e
        self.n_head = n_head
        
        self.bn1 = nn.BatchNorm1d(self.dim_e)
        self.bn2 = nn.BatchNorm1d(self.dim_v)
        
        dim_t = 2*self.dim_v+self.dim_e
        self.phi_e = nn.Sequential(nn.Linear(dim_t,self.dim_e),
                                   nn.LeakyReLU(0.2),
                                   nn.Linear(self.dim_e,self.dim_e),
                                   nn.LeakyReLU(0.2),
                                   nn.Linear(self.dim_e,self.dim_e))
        
        self.FCNNa = nn.Sequential(nn.Linear(dim_t,self.dim_v),
                                   nn.LeakyReLU(0.2),
                                   nn.Linear(self.dim_v,self.dim_v*self.n_head))
        
        self.FCNNm = nn.Sequential(nn.Linear(dim_t,self.dim_v),
                                   nn.LeakyReLU(0.2),
                                   nn.Linear(self.dim_v,self.dim_v*self.n_head))
        
        self.Wout = nn.Linear(self.dim_v*self.n_head,self.dim_v)

    def forward(self,node_fea,idx1,idx2,edge_fea):
        node1 = node_fea[idx1]
        node2 = node_fea[idx2]
        
        #Update edge feature
        z = torch.cat([node1,node2,edge_fea],dim=1)
        ek_prime = self.bn1(self.phi_e(z))
        edge_new = edge_fea + ek_prime
        
        #Get attention vector
        Nt = len(edge_fea)
        z = torch.cat([node1,node2,ek_prime],dim=1)
        sij = self.FCNNa(z).view(Nt,self.n_head,self.dim_v)
        aij = scatter_softmax(sij,idx1,dim=0)
        
        #Update node feature
        B = len(node_fea)
        F = self.n_head*self.dim_v
        mij = self.FCNNm(z).view(Nt,self.n_head,self.dim_v)
        msg_ = aij*mij
        msg = scatter_sum(msg_.view(Nt,F),idx1,dim=0,out=torch.zeros(B,F).cuda())
        if self.n_head > 1:
            msg = self.Wout(msg)
        node_new = node_fea + self.bn2(msg)
        return node_new,edge_new
    
class MPNN_A(nn.Module):
    def __init__(self,dim_v0,dim_v1,dim_e0,dim_e1,n_attn,dim_h,n_h,n_head):
        super(MPNN_A,self).__init__()
        self.v_emb = nn.Embedding(dim_v0,dim_v1)
        self.e_emb = nn.Linear(dim_e0,dim_e1)
        
        self.attns = nn.ModuleList([AttentionHead(dim_v1,dim_e1,n_head) for _ in range(n_attn)])
        
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
        
        for attn in self.attns:
            node_fea,edge_fea = attn(node_fea,idx1,idx2,edge_fea)
         
        vi_e_bar = scatter_mean(edge_fea,idx1,dim=0,out=torch.zeros_like(node_fea))
            
        crys_fea = torch.cat([vi_e_bar,node_fea],dim=1)
        crys_fea = scatter_mean(crys_fea,idx3,dim=0)
        crys_fea = self.conv_to_fc_f(self.conv_to_fc(crys_fea))
        
        if hasattr(self,'fcs'):
            for fc,f in zip(self.fcs,self.fs):
                crys_fea = f(fc(crys_fea))
                
        out = self.fc_out(crys_fea)
        return out
    
