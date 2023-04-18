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
mapper = pickle.load(open('oc20_data_mapping.pkl','rb'))

ads_idx = [[],[],[],[]]
slab_idx = [[],[],[],[]]

for i,sid in enumerate(sids):
    mm = mapper[sid]
    
    slab_id = mm['class']
    
    ads = mm['ads_symbols']
    if 'N' in ads:
        ads_id = 3
    elif ads.count('C') == 2:
        ads_id = 2
    elif ads.count('C') == 1:
        ads_id = 1
    else:
        ads_id = 0
        
    ads_idx[ads_id].append(i)
    slab_idx[slab_id].append(i)

print('Table 3. Test set performance per slab type')
print('Slab type, Screened ratio, TPR, Precision, F1-score, AUROC')
slab_type = ['Intermetallics','Metalloids','Non-metals','Halides']
for st,ii in zip(slab_type,slab_idx):
    m = GetMetric(T[ii],S[ii],cv=0.5)
    b = GetMetric(T[ii],S[ii],cv=0.0)
    
    print(st,m)
    print('Baseline',b)
    print('------------------------------')
 
print('Table 4. Test set performance per adsorbate type')
print('Ads type, Screened ratio, TPR, Precision, F1-score, AUROC')
ads_type = ['O/H only','C1','C2','N-included']
for st,ii in zip(ads_type,ads_idx):
    m = GetMetric(T[ii],S[ii],cv=0.5)
    b = GetMetric(T[ii],S[ii],cv=0.0)
    
    print(st,m)
    print('Baseline',b)
    print('------------------------------')