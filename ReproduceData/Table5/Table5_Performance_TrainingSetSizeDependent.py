# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 14:07:46 2023

@author: User
"""

import numpy as np
import seaborn as sns
import pickle

from sklearn.metrics import roc_auc_score

sns.set_style('whitegrid')

def GetMetric(T,S,cv):
    tpr = np.sum(S[T==1]>=cv)/np.sum(T==1)
    prec = np.sum(T[S>=cv]==1)/np.sum(S>=cv)
    sr = np.sum(S>=cv)/len(T)
    f1 = 2*tpr*prec/(tpr+prec)
    auc = roc_auc_score(T,S)
    return [sr,tpr,prec,f1,auc]

metrics = [0,0,0,0,0,0,0]
for ss in [1111,2222,3333,4444,5555]: 
    #average over five independent trials (trained with different random seeds)
    name = 'RandomEnsScores/Ensemble_'+str(ss)+'.pkl'
    ee = pickle.load(open(name,'rb'))
    
    x = ee['Score'][:6]
    t = ee['True']
    
    for i in range(6):
        mi = GetMetric(t,x[i],cv=0.5)
        metrics[i] += np.array(mi)/5 
        
    if ss == 1111:
        whole = ee['Score'][-1]
        mi = GetMetric(t,whole,cv=0.5)
        metrics[-1] += np.array(mi)
       
NumDat = [10000,23016,46032,115082,230164,345246,460328]
Max = 460328
for i,nd in enumerate(NumDat):
    ri = round(nd/Max,2)
    print('Training set size, Screened ratio, TPR, Precision, F1-score, AUROC')
    print([ri]+[round(v,3) for v in metrics[i]])  
    if i > 0:
        RI = (metrics[i]-metrics[0])/metrics[0]
        print('Relative improvement TPR/Precision/F1/AUROC vs. minimum training set size')
        print([ri]+[round(v*100,1) for v in RI[1:]])
    print('-----------------------------------------')

