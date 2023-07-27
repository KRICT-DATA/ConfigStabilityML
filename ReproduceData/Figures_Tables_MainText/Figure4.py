# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 15:29:27 2023

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

plt.figure(dpi=200)
mapper = pickle.load(open('../oc20_data_mapping.pkl','rb'))
ccs = ['#006699','#6699cc','#ff6600','#ffcc00']
tts = ['(a) Positive','(b) Adsorbate dissociation, AS','(c) Adsorbate desorption, AD','(d) Surface reconstruction, SR']
Labs = ['ID','OOD-Cat','OOD-Ads','OOD-Both']
vids = ['id','ood_cat','ood_ads','ood_both']
for vi in [0,1,2,3]:
    P,T,slabs,aas,sids = GetTestData('TestSetPredictions_460328.pkl',[vids[vi]])
    P = np.mean(P,0)
    for a in [0,1,2,3]:
        plt.subplot(1,4,a+1)
        sns.kdeplot(P[aas==a],color=ccs[vi],alpha=0.7,label=Labs[vi])
        plt.title(tts[a],fontsize=13)
        plt.xlabel('Classification score')
        
plt.subplot(1,4,1)
plt.legend(loc='best',fontsize=11)