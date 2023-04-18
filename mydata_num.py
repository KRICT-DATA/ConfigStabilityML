# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 09:19:05 2022

@author: User
"""


import pickle
import lmdb
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate

def collate_pool(dataset_list):
    batch_num = []
    batch_edge_idx,batch_eij = [],[]
    batch_y = []
    batch_idx,batch_sid = [],[]
    
    base_idx = 0
    for i,(num,edge_idx,eij,y,sid) in enumerate(dataset_list):
        n = len(num)
                
        batch_num.append(num)
        batch_edge_idx.append(edge_idx+base_idx)
        batch_eij.append(eij)
        batch_y.append(y)
        batch_idx += [i]*n
        
        base_idx += n
        batch_sid.append(sid)
        
    batch_num = torch.cat(batch_num,dim=0)
    batch_edge_idx = torch.cat(batch_edge_idx,dim=1)
    batch_eij = torch.cat(batch_eij,dim=0)
    batch_y = torch.cat(batch_y,dim=0).view(-1)
    batch_idx = torch.tensor(batch_idx)
    
    return batch_num,batch_edge_idx,batch_eij,batch_y,batch_idx,batch_sid
        
class SurfaceData(Dataset):
    def __init__(self,map_dir,lmdb_dir,dim_edge,radius=6,dmin=0,is_train=False,num_dat=10000,random_seed=1111,is_test=False):
        self.map_dir = map_dir
        self.lmdb_dir = lmdb_dir
        self.num_dat = num_dat
        self.is_test = is_test

        #edge related
        self.radius = radius
        self.dmin = dmin
        self.dim_edge = dim_edge
        self.mu = torch.Tensor(self.even_samples(self.dmin,self.radius,self.dim_edge)).view(1,-1)
        
        #get data mapper
        self.mapper = pickle.load(open(self.map_dir,'rb'))
        
        #get data
        self.env = lmdb.open(self.lmdb_dir+'/data.lmdb',subdir=False,readonly=True,lock=False,readahead=False,meminit=False,max_readers=1)       
 
        if is_train:
            tmp = [f"{j}".encode('ascii') for j in range(self.env.stat()['entries'])]
            np.random.seed(random_seed)     
            np.random.shuffle(tmp)
            self._keys = tmp[:self.num_dat]
        else:
            self._keys = [f"{j}".encode('ascii') for j in range(self.env.stat()['entries'])]
               
        self._geos = [pickle.loads(self.env.begin().get(_key)).__dict__ for _key in tqdm(self._keys)]

        np.random.seed(1234)
        np.random.shuffle(self._geos)
        
    def even_samples(self,min,max,n_samples):
        samples = np.empty(n_samples)
        len = (max-min)/n_samples
        for i in range(n_samples):
            samples[i] = min + len*(i+1)
        return samples
    
    def __len__(self):
        return len(self._keys)
    
    def __getitem__(self,idx):
        dat = self._geos[idx]
        
        sid = 'random'+str(dat['sid'])
        a = self.mapper[sid]['anomaly']
        if a == 0: #normal data
            y = torch.tensor([1])
        else: #anomaly
            y = torch.tensor([0])
        
        nums = dat['atomic_numbers'].int()
        
        pos_relaxed = dat['pos']
        edge_index = dat['edge_index']
        
        cell_offsets = dat['cell_offsets'].float()
        cell = dat['cell'].squeeze(0)

        if self.is_test:
            ri = pos_relaxed[edge_index[1]]
            rj = pos_relaxed[edge_index[0]] + torch.mm(cell_offsets,cell)
            dij = torch.sqrt(torch.sum((ri-rj)**2,1)).unsqueeze(1)
        else:
            dij = dat['distances'].unsqueeze(1)
        eij = torch.exp(-(dij-self.mu)**2)
        
        return nums,edge_index,eij,y,sid
        
