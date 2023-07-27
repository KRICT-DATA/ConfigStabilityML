# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 15:36:06 2023

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

sns.set_style('whitegrid')

plt.rcParams.update({'font.size': 13})
plt.rcParams['axes.linewidth'] = 2.0
plt.rcParams['xtick.major.width'] = 2.0
plt.rcParams['ytick.major.width'] = 2.0
plt.rcParams['xtick.minor.width'] = 2.0
plt.rcParams['ytick.minor.width'] = 2.0

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

vids = ['id','ood_cat','ood_ads','ood_both']
mapper = pickle.load(open('../oc20_data_mapping.pkl','rb'))
P1,T1,slabs1,aas1,sids1 = GetTestData('TestSetPredictions_460328.pkl',vids[:2])
P2,T2,slabs2,aas2,sids2 = GetTestData('TestSetPredictions_460328.pkl',vids[2:])

E1 = np.mean(P1,0)
U1 = np.sqrt(E1*(1-E1))

E2 = np.mean(P2,0)
U2 = np.sqrt(E2*(1-E2))

name_slab = ['(a) Intermetallics','(b) Metalloids','(c) Non-metals','(d) Halides']
plt.figure(dpi=200)
for i in range(4):
    plt.subplot(1,4,i+1)
    sns.kdeplot(U1[slabs1==i],color='#6699cc',label='Seen ads.')
    sns.kdeplot(U2[slabs2==i],color='#ffcc00',label='Unseen ads.')
    plt.title(name_slab[i],fontsize=13)
    plt.xlabel('Total uncertainty')
    
    if i == 0:
        plt.legend(loc='best',fontsize=11)