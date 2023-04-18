# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 14:01:30 2022

@author: User
"""

import sys
sys.path.append('./')

import os
import numpy as np
import pickle

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim

from mydata_num import SurfaceData,collate_pool
from utils_train import *

from sklearn.metrics import roc_auc_score,roc_curve

def Train(classifier,loader,loss_func,optim_func):
	classifier.train()
	L = 0 
	N = 0
	for inp in tqdm(loader):
		num = inp[0].cuda()
		idx1,idx2 = inp[1][1].cuda(),inp[1][0].cuda()
		eij = inp[2].cuda()
		y = inp[3].cuda()
		agg_idx = inp[4].cuda()

		out = classifier(num,eij,idx1,idx2,agg_idx)
		loss = loss_func(out,y.view(-1))

		optim_func.zero_grad()
		loss.backward()
		optim_func.step()

		L += loss.item()
		N += len(num)
	return [L/N],optim_func

def Validation(classifier,loader,loss_func):
	classifier.eval()
	L = 0
	N = 0
	T = []
	S = []
	for inp in tqdm(loader):
		num = inp[0].cuda()
		idx1,idx2 = inp[1][1].cuda(),inp[1][0].cuda()
		eij = inp[2].cuda()
		y = inp[3].cuda()
		agg_idx = inp[4].cuda()

		out = classifier(num,eij,idx1,idx2,agg_idx)
		loss = loss_func(out,y.view(-1))

		L += loss.item()
		N += len(num)

		T.append(y.cpu().detach().numpy().reshape(-1,1))
		S.append(out.cpu().detach().numpy().reshape(-1,2))

	T = np.vstack(T).flatten()
	S = np.vstack(S).reshape(-1,2)

	C_ = np.max(S,1).reshape(-1,1)
	exp_a = np.exp(S-C_)
	sum_exp_a = np.sum(exp_a,1).reshape(-1,1)
	S = exp_a/sum_exp_a

	metric = GetClfMetric(T,S)
	return [L/N]+metric

def GetClfMetric(T,S):
	AUC = roc_auc_score(T,S[:,1])

	P = np.argmax(S,1)
	TPR = np.sum(P[T==1]==1)/np.sum(T==1)
	PREC = np.sum(T[P==1]==1)/np.sum(P==1)
	SR = np.sum(P==1)/len(P)
	F1 = 2*TPR*PREC/(TPR+PREC)
	return [TPR,PREC,SR,AUC,F1]

### Arguments ###
model_type = sys.argv[1] #CGCNN,SchNet,MPNN,MPNN_A,Matformer
num_dat = int(sys.argv[2])
random_seed = int(sys.argv[3])
#################

### Model save related ###
saved_path = 'RandomSeed_'+str(random_seed)+'/'+model_type
if not os.path.isdir(saved_path):
	os.makedirs(saved_path,exist_ok=True)
##########################

### Dataset related ###
map_dir = 'oc20_data_mapping.pkl'
train_dir = 'data/is2re/all/train/'
val_set = {'id':'data/is2re/all/val_id/',
           'ood_ads':'data/is2re/all/val_ood_ads/',
           'ood_cat':'data/is2re/all/val_ood_cat/',
           'ood_both':'data/is2re/all/val_ood_both/'}
#######################

### Load model and parameters ###
model_func = {'CGCNN':LoadCGCNN,'SchNet':LoadSchNet,'MPNN':LoadMPNN,'MPNN_A':LoadMPNN_A,'Matformer':LoadMatformer}
model,model_params = model_func[model_type](num_dat,random_seed)
radius = model_params['radius']
batch_size = model_params['batch_size']
lr = model_params['lr']
if model_type == 'CGCNN':
	dim_e0 = model_params['dim_edge']
else:
	dim_e0 = model_params['dim_e0']
#################################

### Load dataloader ###
train_data = SurfaceData(map_dir,train_dir,dim_edge=dim_e0,radius=radius,dmin=0,is_train=True,num_dat=num_dat,random_seed=random_seed)
train_loader = DataLoader(train_data,batch_size=batch_size,collate_fn=collate_pool,shuffle=True)

ValNames = ['id','ood_ads','ood_cat','ood_both']
ValLoaders = []
for val_name in tqdm(ValNames):
	val_data = SurfaceData(map_dir,val_set[val_name],dim_edge=dim_e0,radius=radius,dmin=0,random_seed=random_seed)
	val_loader = DataLoader(val_data,batch_size=batch_size,collate_fn=collate_pool)

	ValLoaders.append(val_loader)
#######################

### Training & validation ###
ce_loss = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=lr)

LOGs = [['Epoch,ValName,[TPR,PREC,SR,AUC,F1]\n']]*4
AUC_BESTs = [-10000000]*4
for epoch in range(50):
	tr_loss,optimizer = Train(model,train_loader,ce_loss,optimizer)

	with torch.no_grad():
		for i,val_loader in enumerate(ValLoaders):
			savepref = saved_path+'/Model_'+str(num_dat)+'_'+ValNames[i]

			val_loss = Validation(model,val_loader,ce_loss)
			auc_ = val_loss[-2]

			ll = ','.join([str(i) for i in tr_loss+val_loss])+'\n'
			LOGs[i].append(ll)
			open(savepref+'.txt','w').writelines(LOGs[i])

			print(epoch,ValNames[i],[round(v,3) for v in val_loss[1:]])

			if auc_ > AUC_BESTs[i]:
				chkpt_dic = {'epoch':epoch,'state_dict':model.state_dict(),'model_params':model_params}
				torch.save(chkpt_dic,savepref+'.best.pth.tar')
				AUC_BESTs[i] = auc_
#############################
