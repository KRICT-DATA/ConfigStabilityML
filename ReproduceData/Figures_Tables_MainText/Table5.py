# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 12:40:39 2023

@author: User
"""

import numpy as np
import pickle
from tqdm import tqdm
from collections import defaultdict
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score,f1_score
from sklearn.metrics import matthews_corrcoef

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import seaborn as sns

def GetData(name):
    data = pickle.load(open(name,'rb'))
    Ps = []
    for i in range(5):
        tmp = []
        tmp2 = []
        T = []
        for ood in ['id','ood_cat','ood_ads','ood_both']:
            Pi = data['Trial_'+str(i)][ood]['S']
            tmp.append(Pi)
            T.append(data['Trial_'+str(i)][ood]['T'].reshape(-1,1))
            
        tmp = np.vstack(tmp)
        Ps.append(tmp.reshape(1,-1,5))
        T = np.vstack(T).flatten()
        
    Ps = np.vstack(Ps)
    return Ps,T

def GetMetric(T,S,cv):
    L = 1*(S>=cv)
    tpr = np.sum(S[T==1]>=cv)/np.sum(T==1)
    prec = np.sum(T[S>=cv]==1)/np.sum(S>=cv)
    f1 = f1_score(T,L)
    auc = roc_auc_score(T,S)
    mcc = matthews_corrcoef(T,L)
    return [tpr,prec,f1,auc,mcc]

mapper = pickle.load(open('oc20_data_mapping.pkl','rb'))

out = defaultdict(list)

for n in [10000,23016,46032,115082,230164,345246,460328]:
    name = 'TestSetPredictions_'+str(n)+'.pkl'
    Ps,T = GetData(name)
    
    Ps = np.mean(Ps,-1)
    
    m = list([GetMetric(T,Ps[j],0.5) for j in range(5)])
    mu = np.mean(m,0)
    out[n] = mu
    
print('Training set size, TPR, Precision, F1-score, AUROC, MCC')
print('-----------------------------------------')
Nmax = 460328
m0 = out[10000]
for n in [10000,23016,46032,115082,230164,345246,460328]:
    mv = out[n]
    if not n == 10000:
        RI = 100*(mv-m0)/m0
        mv = np.round(mv,3).tolist()
        RI = np.round(RI,1).tolist()
    
        rr = [str(n)+' ('+str(round(100*n/Nmax,0))+')']
        ll = [str(v1)+' ('+str(v2)+')' for v1,v2 in zip(mv,RI)]
        print(','.join(rr+ll))
        print('-----------------------------------------')
    else:
        mv = np.round(mv,3).tolist()

        rr = [str(n)+' ('+str(round(100*n/Nmax,0))+')']
        ll = [str(v1) for v1 in mv]
        print(','.join(rr+ll))
        print('-----------------------------------------')
    