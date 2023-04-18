# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 15:10:34 2023

@author: User
"""

import numpy as np
import pickle

from sklearn.metrics import roc_auc_score

def GetMetric(T,S,cv):
    tpr = np.sum(S[T==1]>=cv)/np.sum(T==1)
    prec = np.sum(T[S>=cv]==1)/np.sum(S>=cv)
    sr = np.sum(S>=cv)/len(T)
    f1 = 2*tpr*prec/(tpr+prec)
    auc = roc_auc_score(T,S)
    m = [sr,tpr,prec,f1,auc]
    return [round(v,3) for v in m]


print('Table 1. Test set benchmark results')
print('Model name, Screened ratio, TPR, Precision, F1-score, AUROC')
Models = ['CGCNN','SchNet','MPNN','MPNN-A','Matformer','Ensemble']
BaseName = 'TestSetScore_MM_w_Sids.pkl'

for mm in Models:
    name = BaseName.replace('MM',mm)
    data = pickle.load(open(name,'rb'))
    
    S = data['Score']
    T = data['True']
    
    if mm == 'CGCNN':
        b = GetMetric(T,S,cv=0.0)
        print('Baseline',b)
    
    b = GetMetric(T,S,cv=0.5)
    print(mm,b)
        
    
    