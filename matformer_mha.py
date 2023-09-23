# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 13:32:49 2022

@author: User
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_sum,scatter_mean

class AttentionHead(nn.Module):
    def __init__(self,dim_v,dim_e,dim_h):
        super(AttentionHead,self).__init__()
        self.dim_v = dim_v
        self.dim_e = dim_e
        self.dim_h = dim_h
        
        self.LN_Q = nn.Linear(dim_v,dim_h)
        self.LN_K = nn.Linear(dim_v,dim_h)
        self.LN_E = nn.Linear(dim_e,dim_h)

        self.LN_V = nn.Linear(dim_v,dim_h)
        self.LNorm1 = nn.LayerNorm(3*dim_h)
        self.LN_update = nn.Linear(3*dim_h,3*dim_h)
        self.sigmoid = nn.Sigmoid()
        
        self.LN_msg = nn.Linear(3*dim_h,dim_v)
        self.LNorm2 = nn.LayerNorm(dim_v)
        
        
    def forward(self,node_fea,idx1,idx2,edge_fea):
        Q = self.LN_Q(node_fea)
        K = self.LN_K(node_fea)
        V = self.LN_V(node_fea)
        eij_prime = self.LN_E(edge_fea)
        
        #First step, compute qij, kij and aij;
        qij = torch.cat([Q[idx1],Q[idx1],Q[idx1]],dim=1)
        kij = torch.cat([K[idx1],K[idx2],eij_prime],dim=1)

        d_kij = kij.shape[1]
        aij = (qij*kij)/np.sqrt(d_kij) #(B,3*dim_h)
        
        #Second step, compute message of eij, mij
        zij = torch.cat([V[idx1],V[idx2],eij_prime],dim=1)
        mij_1 = self.sigmoid(self.LNorm1(aij))
        mij_2 = self.LN_update(zij)
        mij = mij_1 * mij_2
        
        #Third step
        msg = self.LNorm2(self.LN_msg(mij))
        mi = scatter_sum(msg,idx1,dim=0,out=torch.zeros_like(node_fea))
        #node_new = self.LN_fea(node_fea) + self.sigma(self.BN(mi))
        return mi

class MultiHeads(nn.Module):
    def __init__(self,dim_v1,dim_e1,dim_h,n_head):
        super(MultiHeads,self).__init__()
        self.n_head = n_head
        self.heads = nn.ModuleList([AttentionHead(dim_v1,dim_e1,dim_h) for _ in range(n_head)])
        self.Wo = nn.Linear(dim_v1*n_head,dim_v1)
        
        self.BN = nn.BatchNorm1d(dim_v1)
        self.sigma = nn.Softplus()
        self.LN_fea = nn.Linear(dim_v1,dim_v1)
        
    def forward(self,node_fea,idx1,idx2,edge_fea):
        mi = []
        
        for single_head in self.heads:
            mi_s = single_head(node_fea,idx1,idx2,edge_fea)
            mi.append(mi_s)
        
        mi = torch.cat(mi,dim=1)

        if self.n_head > 1:
             mi = self.Wo(mi)

        x_new = self.LN_fea(node_fea) + self.sigma(self.BN(mi))
        return x_new

class Matformer(nn.Module):
    def __init__(self,dim_v0,dim_v1,dim_e0,dim_e1,n_attn,dim_h,n_head):
        super(Matformer,self).__init__()
        self.v_emb = nn.Embedding(dim_v0,dim_v1)
        self.e_emb = nn.Sequential(nn.Linear(dim_e0,dim_e1),
                                   nn.Softplus(),
                                   nn.Linear(dim_e1,dim_e1))
        
        self.attns = nn.ModuleList([MultiHeads(dim_v1,dim_e1,dim_h,n_head) for _ in range(n_attn)])
        
        self.conv_to_fc = nn.Linear(dim_h,2*dim_h)
        self.conv_to_fc_f = nn.LeakyReLU(0.2)

        self.fcs = nn.Sequential(nn.Linear(2*dim_h,2*dim_h),
                                 nn.SiLU(),
                                 nn.Linear(2*dim_h,2))

    def forward(self,node_fea,edge_fea,idx1,idx2,idx3):
        node_fea = self.v_emb(node_fea)
        edge_fea = self.e_emb(edge_fea)
        
        for multi_heads in self.attns: #single head
            node_fea = multi_heads(node_fea,idx1,idx2,edge_fea)
        
        crys_fea = scatter_mean(node_fea,idx3,dim=0)
        crys_fea = self.conv_to_fc_f(self.conv_to_fc(crys_fea))
                
        out = self.fcs(crys_fea)
        return out
