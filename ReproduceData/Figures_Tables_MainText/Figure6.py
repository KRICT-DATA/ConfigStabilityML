# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 16:04:52 2023

@author: User
"""

import mpltern
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

sns.set_style('whitegrid')

plt.rcParams.update({'font.size': 13})
plt.rcParams['axes.linewidth'] = 2.0
plt.rcParams['xtick.major.width'] = 2.0
plt.rcParams['ytick.major.width'] = 2.0
plt.rcParams['xtick.minor.width'] = 2.0
plt.rcParams['ytick.minor.width'] = 2.0

def GetMetric(T,S,cv):
    L = 1*(S>=cv)
    tpr = np.sum(S[T==1]>=cv)/np.sum(T==1)
    prec = np.sum(T[S>=cv]==1)/np.sum(S>=cv)
    f1 = f1_score(T,L)
    auc = roc_auc_score(T,S)
    mcc = matthews_corrcoef(T,L)
    rr = np.sum(T==1)/len(T)
    fnr = np.sum(S[T==0]>=cv)/np.sum(T==0)
    return [rr,fnr,tpr,prec,f1,auc,mcc]

def AdsSize(sym):
    aa_ = sym.replace('*','')
    aa = Composition(aa_)

    n = 0
    cc = []
    for k in ['C','O','N']:
        if k in aa:
            n += aa[k]
            cc.append(k)
    return n

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

def GetAds(sid):
    aa_ = mapper[sid]['ads_symbols'].replace('*','')
    aa = Composition(aa_)
    
    n = 0
    cc = []
    for k in ['C','O','N']:
        if k in aa:
            n += aa[k]
            cc.append(k)
     
    kk = AdsType(sid)
    y = mapper[sid]['anomaly']
    
    if y == 4:
        y = 1
    
    return kk,n,y,mapper[sid]['ads_symbols']

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

ccs = ['#006699','#6699cc','#ff6600','#ffcc00']
vids = ['id','ood_cat','ood_ads','ood_both']
mapper = pickle.load(open('oc20_data_mapping.pkl','rb'))

P1,T1,slabs1,aas1,sids1 = GetTestData('TestSetPredictions_460328.pkl',vids)
syms1 = np.array([GetAds(sid)[-1] for sid in sids1])
uni1 = list(set(syms1))

main_typ = {}
for uu in uni1:
    zz = list(set(T1[syms1==uu]))
    if len(zz) == 1:
        print(uu)
        continue
    
    p1 = P1[:,syms1==uu]
    t1 = T1[syms1==uu]
    a1 = aas1[syms1==uu]
    
    rr = []
    nan_chk = False
    for k in [1,2,3]:
        ff = [np.sum(a1==k)/np.sum(t1==0) for j in range(5)]
        ff = np.mean(ff)
        
        rr.append(ff)
    
        if np.isnan(ff):
            nan_chk = True
    
    if nan_chk:
        continue
    
    idx = np.argmax(rr)
    n = AdsSize(uu)
    
    main_typ[uu] = [idx,n]
    
df = defaultdict(list)
for kk in main_typ:
    idx,n = main_typ[kk]
    
    p1 = P1[:,syms1==kk]
    t1 = T1[syms1==kk]
    m = [GetMetric(t1,p1[j],0.5) for j in range(5)]
    m = np.mean(m,0).tolist()+[n]
    
    df[idx].append(m)

plt.figure(dpi=200)
plt.subplot(1,5,1)
vec1 = [np.array(df[idx])[:,-1] for idx in [0,1,2]]
plt.violinplot(vec1)
plt.title('(a) The number of non-H atoms',fontsize=13)
plt.xticks([1,2,3],['AS-main','AD-main','SR-main'])

plt.subplot(1,5,2)
vec1 = [np.array(df[idx])[:,0] for idx in [0,1,2]]
plt.violinplot(vec1)
plt.title('(b) Prior positive ratio',fontsize=13)
plt.xticks([1,2,3],['AS-main','AD-main','SR-main'])
    
plt.subplot(1,5,3)
vec1 = [np.array(df[idx])[:,-4] for idx in [0,1,2]]
plt.violinplot(vec1)
plt.title('(c) F1-score',fontsize=13)
plt.xticks([1,2,3],['AS-main','AD-main','SR-main'])

plt.subplot(1,5,4)
vec1 = [np.array(df[idx])[:,-3] for idx in [0,1,2]]
plt.violinplot(vec1)
plt.title('(d) AUROC',fontsize=13)
plt.xticks([1,2,3],['AS-main','AD-main','SR-main'])

plt.subplot(1,5,5)
vec1 = [np.array(df[idx])[:,-2] for idx in [0,1,2]]
plt.violinplot(vec1)
plt.title('(e) MCC',fontsize=13)
plt.xticks([1,2,3],['AS-main','AD-main','SR-main'])
