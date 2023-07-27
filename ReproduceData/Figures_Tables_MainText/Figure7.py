# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 11:44:11 2023

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

plt.rcParams.update({'font.size': 11})
plt.rcParams['axes.linewidth'] = 2.0
plt.rcParams['xtick.major.width'] = 2.0
plt.rcParams['ytick.major.width'] = 2.0
plt.rcParams['xtick.minor.width'] = 2.0
plt.rcParams['ytick.minor.width'] = 2.0

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

def set_labels(ax):
    """Set ternary-axis labels."""
    ax.set_tlabel('AS')
    ax.set_llabel('AD')
    ax.set_rlabel('SR')
    ax.taxis.set_label_position("tick1")
    ax.laxis.set_label_position("tick1")
    ax.raxis.set_label_position("tick1")
    
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

P2,T2,slabs2,aas2,sids2 = GetTestData('TestSetPredictions_460328.pkl',vids)
syms2 = np.array([GetAds(sid)[-1] for sid in sids2])

uni2 = list(set(syms2.tolist()))
size_uni2 = np.array([AdsSize(sym) for sym in uni2])

df = defaultdict(list)
df2 = []
for sym,n in zip(uni2,size_uni2):
    pp = P2[:,syms2==sym]
    tt = T2[syms2==sym]
    aa = aas2[syms2==sym]
    r = np.sum(aa==0)/len(aa)
    tmp = []
    for A in [1,2,3]:
        vv = np.sum(pp[:,aa==A]>=0.5,1)/np.sum(pp[:,tt==0]>=0.5,1)
        tmp.append(np.mean(vv))
    if np.sum(tmp) >= 0:
        df2.append(tmp+[r])
        
df2 = np.array(df2)

plt.figure(dpi=200)
ax = plt.subplot(projection="ternary")
pc = ax.scatter(df2[:,0],df2[:,1],df2[:,2],c=df2[:,3],s=40,cmap='RdYlBu',alpha=1.0,vmin=0,vmax=1,edgecolors='k')
ax.set_xlim(-0.6883961864829182,0.6909407918266515)
ax.set_ylim(-0.07350897575056309,1.1210318878447898)

cax = ax.inset_axes([1.05, 0.1, 0.05, 0.9], transform=ax.transAxes)
colorbar = fig.colorbar(pc, cax=cax)
colorbar.set_label("Prior positive ratio", rotation=270, va="baseline")

set_labels(ax)
