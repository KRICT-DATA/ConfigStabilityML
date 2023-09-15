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
labs = ['ratio=0.05','ratio=0.10','ratio=0.25','ratio=1.00']
nums = [23016,46032,115082,460328]

plt.figure(dpi=200)
for i in range(4):
    n = nums[i]
    name = 'TestSetPredictions_'+str(n)+'.pkl'
    Ps,T = GetData(name)
    Ps = np.mean(np.mean(Ps,-1),0)
    
    plt.subplot(1,2,1)
    sns.kdeplot(Ps[T==1],alpha=0.6)
    plt.xlabel('Classification score')
    plt.title('(a) Positive class',fontsize=13)
    
    plt.subplot(1,2,2)
    sns.kdeplot(Ps[T==0],alpha=0.6,label=labs[i])
    plt.xlabel('Classification score')
    plt.title('(b) Negative class',fontsize=13)

plt.subplot(1,2,2)
plt.legend(loc='best',fontsize=11)
