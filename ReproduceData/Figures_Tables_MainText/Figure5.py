# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 08:16:14 2023

@author: User
"""

import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

from collections import defaultdict
from pymatgen.core import Composition

from scipy.stats import pearsonr

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score,f1_score
from sklearn.metrics import matthews_corrcoef

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

sns.set_style('whitegrid')

plt.rcParams.update({'font.size': 13})
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

sids = pickle.load(open('TrainingSetSids.pkl','rb'))

ValID = ['id','ood_ads','ood_cat','ood_both']
for vv in ValID:
    sids += pickle.load(open('ValSetSids.'+vv+'.pkl','rb'))
    sids += pickle.load(open('TestSetSids.'+vv+'.pkl','rb'))

mapper = pickle.load(open('oc20_data_mapping.pkl','rb'))

df = defaultdict(list)

for i in range(len(sids)):
    sid = sids[i]
    a = mapper[sid]['ads_symbols']
    y = mapper[sid]['anomaly']
    n = AdsSize(a)
    
    if y == 4:
        y = 1
       
    df[a].append(y)
       
df2 = defaultdict(list)
for a in df:
    v = np.array(df[a])
    n = AdsSize(a)

    r = [np.sum(v==0)/len(v)]
    
    for i in [1,2,3]:
        ff = np.sum(v==i)/np.sum(v>0)
        r.append(ff)
   
    if n >= 5:
        n = 5
 
    df2[n].append(r)
    
kks = np.sort(list(df2.keys()))
tts = ['(a) Positive class','(b) Adsorbate dissociation, AS','(d) Adsorbate desorption, AD','(c) Surface reconstruction, SR']
#tts = ['(a) Positive class','(b) Adsorbate dissociation, AS','(c) Surface reconstruction, SR','(d) Adsorbate desorption, AD']
plt.figure(dpi=200)
fid = 1
for i in [0,1,3,2]:
    plt.subplot(1,4,fid)
    yy = [np.array(df2[k])[:,i] for k in kks]
    plt.boxplot(yy)
    plt.xticks(np.arange(1,7),['*H','1','2','3','4',r'$\geq5$'])
    plt.title(tts[i],fontsize=13)
    fid += 1
    plt.ylim(-0.03,1.03)
    
    if i == 0:
        plt.ylabel('Ratio')
    else:
        plt.ylabel('Relative ratio within negative class',fontsize=12)
    
for i in [0,1,3,2]:
    X = []
    Y = []
    for kk in kks:
        yv = np.array(df2[kk])[:,i]
        xv = np.array([kk]*len(yv))
        
        X.append(xv.reshape(-1,1))
        Y.append(yv.reshape(-1,1))
        
    X = np.vstack(X).flatten()
    Y = np.vstack(Y).flatten()
    
    print(i,pearsonr(X,Y)[0])
    
#===================================================
sids = pickle.load(open('TrainingSetSids.pkl','rb'))

ValID = ['id','ood_cat']
for vv in ValID:
    sids += pickle.load(open('ValSetSids.'+vv+'.pkl','rb'))
    sids += pickle.load(open('TestSetSids.'+vv+'.pkl','rb'))


df = defaultdict(list)

for i in range(len(sids)):
    sid = sids[i]
    a = mapper[sid]['ads_symbols']
    y = mapper[sid]['anomaly']
    n = AdsSize(a)
    
    if y == 4:
        y = 1
       
    df[a].append(y)
       
df2 = defaultdict(list)
for a in df:
    v = np.array(df[a])
    n = AdsSize(a)

    r = [np.sum(v==0)/len(v)]
    
    for i in [1,2,3]:
        ff = np.sum(v==i)/np.sum(v>0)
        r.append(ff)
   
    if n >= 5:
        n = 5
 
    df2[n].append(r)
    
kks = np.sort(list(df2.keys()))
for kk in kks:
    yv = np.array(df2[kk])
    xv = np.array([kk+1]*len(yv))
    
    fid = 1
    for i in [0,1,3,2]:
        xv2 = xv + 0.04*np.random.randn(len(xv))
        plt.subplot(1,4,fid)
        plt.scatter(xv2,yv[:,i],marker='o',s=10,c='#6699cc')
        fid += 1
        
#===================================================
sids = []

ValID = ['ood_ads','ood_both']
for vv in ValID:
    sids += pickle.load(open('ValSetSids.'+vv+'.pkl','rb'))
    sids += pickle.load(open('TestSetSids.'+vv+'.pkl','rb'))


df = defaultdict(list)

for i in range(len(sids)):
    sid = sids[i]
    a = mapper[sid]['ads_symbols']
    y = mapper[sid]['anomaly']
    n = AdsSize(a)
    
    if y == 4:
        y = 1
       
    df[a].append(y)
       
df2 = defaultdict(list)
for a in df:
    v = np.array(df[a])
    n = AdsSize(a)

    r = [np.sum(v==0)/len(v)]
    
    for i in [1,2,3]:
        ff = np.sum(v==i)/np.sum(v>0)
        r.append(ff)
   
    if n >= 5:
        n = 5
 
    df2[n].append(r)
    
kks = np.sort(list(df2.keys()))
for kk in kks:
    yv = np.array(df2[kk])
    xv = np.array([kk+1]*len(yv))
    
    fid = 1
    for i in [0,1,3,2]:
        xv2 = xv + 0.04*np.random.randn(len(xv))
        plt.subplot(1,4,fid)
        plt.scatter(xv2,yv[:,i],marker='o',s=10,c='#ffcc00')        
        fid += 1
        