# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 01:09:10 2023

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

def AdsType(sid):
    aa = mapper[sid]['ads_symbols']
    if 'N' in aa:
        return 'N'
    elif aa.count('C') == 2:
        return 'C2'
    elif aa.count('C') == 1:
        return 'C1'
    else:
        return 'OH'

def GetTestData(name,vids):
    data = pickle.load(open(name,'rb'))
    Ps = []
    for i in range(5):
        tmp = []
        tmp2 = []
        T = []
        sids = []
        for ood in vids:
            Pi = data['Trial_'+str(i)][ood]['S']
            tmp.append(Pi)
            T.append(data['Trial_'+str(i)][ood]['T'].reshape(-1,1))
            sids += data['Trial_'+str(i)][ood]['sids']
            
        tmp = np.vstack(tmp)
        Ps.append(tmp.reshape(1,-1,5))
        T = np.vstack(T).flatten()
        
    Ps = np.vstack(Ps)
    Ps = np.mean(Ps,-1)
    slabs = np.array([mapper[sid]['class'] for sid in sids])
    
    aas = []
    for sid in sids:
        aa = mapper[sid]['anomaly']
        if aa == 4:
            aa = 1
        aas.append(aa)
    aas = np.array(aas)
    
    return Ps,T,slabs,aas,sids

mapper = pickle.load(open('oc20_data_mapping.pkl','rb'))
Ps,T,slabs,aas,sids = GetTestData('TestSetPredictions_460328.pkl',['id','ood_cat','ood_ads','ood_both'])
ads = np.array([AdsType(sid) for sid in sids])

print('-------------------------------')
print('SlabType,TPR,Precision,F1-score,AUROC,MCC')
print('-------------------------------')
name_slab = ['Intermetallics','Metalloids','Non-metals','Halides']
for s in [0,1,2,3]:
    m = [list(GetMetric(T[slabs==s],Ps[trial,slabs==s],0.5)) for trial in range(5)]
    base = np.round(list(GetMetric(T[slabs==s],1*(Ps[0,slabs==s]>=0),0.0)),3).tolist()

    mu = np.round(np.mean(m,0),3).tolist()
    std = np.round(np.std(m,0),3).tolist()

    vv = [str(v1)+' ('+str(v2)+')' for v1,v2 in zip(mu,std)]
    print(','.join([name_slab[s]]+vv))
    
    vv = [str(v) for v in base]
    print(','.join(['Baseline']+vv))
    
    print('-------------------------------')
    

print('-------------------------------')
print('AdsType,TPR,Precision,F1-score,AUROC,MCC')
print('-------------------------------')
name_ads = ['O/H only','C1','C2','N-included']
for i,s in enumerate(['OH','C1','C2','N']):
    m = [list(GetMetric(T[ads==s],Ps[trial,ads==s],0.5)) for trial in range(5)]
    base = np.round(list(GetMetric(T[ads==s],1*(Ps[0,ads==s]>=0),0.0)),3).tolist()

    mu = np.round(np.mean(m,0),3).tolist()
    std = np.round(np.std(m,0),3).tolist()

    vv = [str(v1)+' ('+str(v2)+')' for v1,v2 in zip(mu,std)]
    print(','.join([name_ads[i]]+vv))
    
    vv = [str(v) for v in base]
    print(','.join(['Baseline']+vv))
    
    print('-------------------------------')