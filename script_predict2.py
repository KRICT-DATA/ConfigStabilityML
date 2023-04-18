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

from mydata_num2 import SurfaceData,collate_pool
from utils_test import *

from sklearn.metrics import roc_auc_score,roc_curve

def Validation(classifier,loader):
	classifier.eval()
	T = []
	S = []
	Tag = []
	for inp in tqdm(loader):
		num = inp[0].cuda()
		idx1,idx2 = inp[1][1].cuda(),inp[1][0].cuda()
		eij = inp[2].cuda()
		y = inp[3].cuda()
		agg_idx = inp[4].cuda()

		out = classifier(num,eij,idx1,idx2,agg_idx)

		T.append(y.cpu().detach().numpy().reshape(-1,1))
		S.append(out.cpu().detach().numpy().reshape(-1,2))
		Tag += sids

	T = np.vstack(T).flatten()
	S = np.vstack(S).reshape(-1,2)

	C_ = np.max(S,1).reshape(-1,1)
	exp_a = np.exp(S-C_)
	sum_exp_a = np.sum(exp_a,1).reshape(-1,1)
	S = exp_a/sum_exp_a

	metric = GetClfMetric(T,S)

	dic = {'T':T,'S':S,'sids':Tag,'Metric':metric}
	return dic

def GetClfMetric(T,S):
	AUC = roc_auc_score(T,S[:,1])

	P = np.argmax(S,1)
	TPR = np.sum(P[T==1]==1)/np.sum(T==1)
	PREC = np.sum(T[P==1]==1)/np.sum(P==1)
	SR = np.sum(P==1)/len(P)
	F1 = 2*TPR*PREC/(TPR+PREC)
	return [TPR,PREC,SR,AUC,F1]

map_dir = 'oc20_data_mapping.pkl'
test_set = {'id':'data/is2re/all/test_id/',
           'ood_ads':'data/is2re/all/test_ood_ads/',
           'ood_cat':'data/is2re/all/test_ood_cat/',
           'ood_both':'data/is2re/all/test_ood_both/'}

val_set = {'id':'data/is2re/all/val_id/',
           'ood_ads':'data/is2re/all/val_ood_ads/',
           'ood_cat':'data/is2re/all/val_ood_cat/',
           'ood_both':'data/is2re/all/val_ood_both/'}

model_func = {'CGCNN':LoadCGCNN,'SchNet':LoadSchNet,'MPNN':LoadMPNN,'MPNN_A':LoadMPNN_A,'Matformer':LoadMatformer}
ood_types = ['id','ood_ads','ood_cat','ood_both']

model_type = sys.argv[1] #CGCNN,SchNet,MPNN,MPNN_A,Matformer
is_test = bool(int(sys.argv[2])) #1 for loading test data, 0 for loading val data
path_pref = sys.argv[3]
saved_path = path_pref+'/'+model_type
num_dat = sys.argv[4]

BASENAME = 'Model_NN_VV.best.pth.tar'
BASENAME2 = 'Predicted_NN_VV.TT.pkl'
batch_size = 12
for ood in ood_types:
	chkpt_name = saved_path+'/'+BASENAME.replace('NN',num_dat).replace('VV',ood)
	model,radius,dim_e0 = model_func[model_type](chkpt_name)

	if is_test:
		lmdb_dir = test_set[ood]
	else:
		lmdb_dir = val_set[ood]

	cat_data = SurfaceData(map_dir,lmdb_dir,dim_edge=dim_e0,radius=radius,dmin=0,is_train=False,is_test=is_test)
	cat_loader = DataLoader(cat_data,batch_size=batch_size,collate_fn=collate_pool)

	with torch.no_grad():
		results = Validation(model,cat_loader)

	if is_test:
		fname = saved_path+'/'+BASENAME2.replace('NN',num_dat).replace('VV',ood).replace('TT','test')
	else:
		fname = saved_path+'/'+BASENAME2.replace('NN',num_dat).replace('VV',ood).replace('TT','val')
	pickle.dump(results,open(fname,'wb'))
	print('SAVED!',model_type,fname)
