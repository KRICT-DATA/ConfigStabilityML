# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 00:51:49 2023

@author: User
"""

import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

from collections import defaultdict
from pymatgen.core import Composition

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score,f1_score
from sklearn.metrics import matthews_corrcoef

def GetMetric(T,P,c):
    L = 1*(P>=c)
    auc = roc_auc_score(T,P)
    f1 = f1_score(T,L)
    prec = np.sum(T[P>=c]==1)/np.sum(P>=c)
    tpr = np.sum(P[T==1]>=c)/np.sum(T==1)
    mcc = matthews_corrcoef(T,L)
    return tpr,prec,f1,auc,mcc

def GetTestData(name,vids):
    data = pickle.load(open(name,'rb'))
    Ps = []
    for i in range(5):
        tmp = []
        tmp2 = []
        T = []
        #sids = []
        for ood in vids:
            Pi = data['Trial_'+str(i)][ood]['S']
            tmp.append(Pi)
            T.append(data['Trial_'+str(i)][ood]['T'].reshape(-1,1))
            #sids += data['Trial_'+str(i)][ood]['sids']
            
        tmp = np.vstack(tmp)
        Ps.append(tmp.reshape(1,-1,5))
        T = np.vstack(T).flatten()
        
    Ps = np.vstack(Ps)
    Ps = np.mean(Ps,-1)
    return Ps,T

print('TestSetType,TPR,Precision,F1-score,AUROC,MCC')
print('-------------------------------')
TestName = ['ID','OOD-Cat','OOD-Ads','OOD-Both']
for nn,vid in zip(TestName,['id','ood_cat','ood_ads','ood_both']):
    Ps,T = GetTestData('TestSetPredictions_460328.pkl',[vid])
    m = [list(GetMetric(T,Ps[trial,:],0.5)) for trial in range(5)]
    base = np.round(list(GetMetric(T,1*(Ps[0,:]>=0),0.0)),3).tolist()
    
    mu = np.round(np.mean(m,0),3).tolist()
    std = np.round(np.std(m,0),3).tolist()

    vv = [str(v1)+' ('+str(v2)+')' for v1,v2 in zip(mu,std)]
    print(','.join([nn]+vv))
    
    vv = [str(v) for v in base]
    print(','.join(['Baseline']+vv))
    print('-------------------------------------')