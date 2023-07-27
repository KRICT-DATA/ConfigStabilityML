# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 14:07:46 2023

@author: User
"""

import numpy as np
import seaborn as sns
import pickle

from sklearn.metrics import roc_auc_score,f1_score
from sklearn.metrics import matthews_corrcoef

sns.set_style('whitegrid')

def GetMetric(T,S,cv):
    L = 1*(S>=cv)
    tpr = np.sum(S[T==1]>=cv)/np.sum(T==1)
    prec = np.sum(T[S>=cv]==1)/np.sum(S>=cv)
    #sr = np.sum(S>=cv)/len(T)
    #f1 = 2*tpr*prec/(tpr+prec)
    f1 = f1_score(T,L)
    auc = roc_auc_score(T,S)
    mcc = matthews_corrcoef(T,L)
    return [tpr,prec,f1,auc,mcc]

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
    print('Training set size, TPR, Precision, F1-score, AUROC, MCC')
    print([ri]+[round(v,3) for v in metrics[i]])  
    if i > 0:
        RI = (metrics[i]-metrics[0])/metrics[0]
        print('Relative improvement TPR/Precision/F1/AUROC/MCC vs. minimum training set size')
        print([ri]+[round(v*100,1) for v in RI])
    print('-----------------------------------------')

Ps,T = GetTestData('TestSetPredictions_460328.pkl',['id','ood_cat','ood_ads','ood_both'])
m = [list(GetMetric(T,Ps[trial,:],0.5)) for trial in range(5)]
mu = np.round(np.mean(m,0),3).tolist()
RI = (mu-metrics[0])/metrics[0]

print('Training set size, TPR, Precision, F1-score, AUROC, MCC')
print([1.0]+[round(v,3) for v in mu])  
print('Relative improvement TPR/Precision/F1/AUROC/MCC vs. minimum training set size')
print([1.0]+[round(v*100,1) for v in RI])
print('-----------------------------------------')