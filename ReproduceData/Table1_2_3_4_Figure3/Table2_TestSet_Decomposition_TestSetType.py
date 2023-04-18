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

data = pickle.load(open('TestSetScore_Ensemble_w_Sids.pkl','rb'))
S = data['Score']
T = data['True']
sids = data['Sids']

TestSetNums = [24948,24930,24965,24985]
DatNums = np.cumsum([0]+TestSetNums)

print('Table 2. Test set performance per test set type')
print('Test set type, Screened ratio, TPR, Precision, F1-score, AUROC')
test_type = ['ID','OOD-Cat','OOD-Ads','OOD-Both']
for ii,st in enumerate(test_type):
    vi = DatNums[ii]
    vj = DatNums[ii+1]
    
    m = GetMetric(T[vi:vj],S[vi:vj],cv=0.5)
    b = GetMetric(T[vi:vj],S[vi:vj],cv=0.0)

    print(st,m)
    print('Baseline',b)
    print('------------------------------')
 