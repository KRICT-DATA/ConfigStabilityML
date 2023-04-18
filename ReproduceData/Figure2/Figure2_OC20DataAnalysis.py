# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 16:14:19 2023

@author: User
"""

import pickle
import numpy as np
from tqdm import tqdm
from pymatgen.core import Composition

import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style('whitegrid')

plt.rcParams.update({'font.size': 12})
plt.rcParams['axes.linewidth'] = 2.0
plt.rcParams['xtick.major.width'] = 2.0
plt.rcParams['ytick.major.width'] = 2.0
plt.rcParams['xtick.minor.width'] = 2.0
plt.rcParams['ytick.minor.width'] = 2.0

def GetStat1(dtyp,mapper):
    if dtyp == 'Train':
        sids = pickle.load(open('TrainingSetSids.pkl','rb'))
    elif dtyp == 'Val':
        ValID = ['id','ood_ads','ood_cat','ood_both']
        sids = []
        for vv in ValID:
            sids += pickle.load(open('ValSetSids.'+vv+'.pkl','rb'))
    else:
        ValID = ['id','ood_ads','ood_cat','ood_both']
        sids = []
        for vv in ValID:
            sids += pickle.load(open('TestSetSids.'+vv+'.pkl','rb'))

    ss = [0,0]
    for sid in sids:
        a = mapper[sid]['anomaly']
        if a == 0:
            ss[0] += 1
        else:
            ss[1] += 1
    return np.array(ss)

def GetStat2(sids,mapper):
    ss = np.zeros((4,2))
    for sid in sids:
        a = mapper[sid]['anomaly']
        c = mapper[sid]['class']
        if a == 0:
            ss[c,0] += 1
        else:
            ss[c,1] += 1
    return np.array(ss)

def GetStat3(sids,mapper):
    ss = np.zeros((4,2))
    for sid in sids:
        a = mapper[sid]['anomaly']
        ads = mapper[sid]['ads_symbols']
        
        if 'N' in ads:
            c = 3
        elif ads.count('C') == 2:
            c = 2
        elif ads.count('C') == 1:
            c = 1
        elif 'O' in ads or 'H' in ads:
            c = 0
        
        if a == 0:
            ss[c,0] += 1
        else:
            ss[c,1] += 1
    return np.array(ss)

mapper = pickle.load(open('oc20_data_mapping.pkl','rb'))

sids = pickle.load(open('TrainingSetSids.pkl','rb'))
ValID = ['id','ood_ads','ood_cat','ood_both']
for vv in ValID:
    sids += pickle.load(open('ValSetSids.'+vv+'.pkl','rb'))
    sids += pickle.load(open('TestSetSids.'+vv+'.pkl','rb'))

ratio1 = GetStat1('Train', mapper)
ratio2 = GetStat1('Val', mapper)
ratio3 = GetStat1('Test', mapper)
ratio4 = ratio1+ratio2+ratio3

X = []
for rr in [ratio4,ratio1,ratio2,ratio3]:
    pp = rr/np.sum(rr)
    X.append(pp.reshape(1,2))
    
plt.figure(dpi=200)
plt.subplot(1,3,1)
X = np.vstack(X)
xv = np.arange(4)
plt.bar(xv-0.1,X[:,0],width=0.2,label='Positive',fc='#6699cc',ec='k')
plt.bar(xv+0.1,X[:,1],width=0.2,label='Negative',fc='gold',ec='k')
plt.xticks(xv,['Whole','Train','Validation','Test'])
plt.ylabel('Ratio of data')
plt.legend(loc='best',fontsize=10,ncol=2)
plt.title('(a) Class ditribution',fontsize=12)
plt.ylim(0,1.02)

_ratio = GetStat2(sids,mapper)
nums = np.sum(_ratio,1)/np.sum(_ratio)
ratio = _ratio/np.sum(_ratio,1).reshape(-1,1)
plt.subplot(2,3,2)
xv = np.arange(4)
plt.bar(xv,nums,width=0.2,fc='#adefd1ff',ec='k')
plt.xticks(xv,['','','',''])
plt.title('(b) Slab type distribution',fontsize=12)
plt.ylim(0,0.52)
plt.ylabel('Ratio of data')

plt.subplot(2,3,5)
xv = np.arange(4)
plt.bar(xv-0.1,ratio[:,0],width=0.2,label='Positive',fc='#6699cc',ec='k')
plt.bar(xv+0.1,ratio[:,1],width=0.2,label='Negative',fc='gold',ec='k')
plt.xticks(xv,['Intermetallics','Metalloids','Non-metals','Halides'],rotation=20)
plt.legend(loc='best',fontsize=10,ncol=2)
plt.title('(c) Class distribution per slab type',fontsize=12)
plt.ylim(0,1.02)
plt.ylabel('Ratio of data')

_ratio = GetStat3(sids,mapper)
nums = np.sum(_ratio,1)/np.sum(_ratio)
ratio = _ratio/np.sum(_ratio,1).reshape(-1,1)
plt.subplot(2,3,3)
xv = np.arange(4)
plt.bar(xv,nums,width=0.2,fc='#adefd1ff',ec='k')
plt.xticks(xv,['','','',''])
plt.title('(d) Adsorbate type distribution',fontsize=12)
plt.ylim(0,0.52)
plt.ylabel('Ratio of data')

plt.subplot(2,3,6)
xv = np.arange(4)
plt.bar(xv-0.1,ratio[:,0],width=0.2,label='Positive',fc='#6699cc',ec='k')
plt.bar(xv+0.1,ratio[:,1],width=0.2,label='Negative',fc='gold',ec='k')
plt.xticks(xv,['O/H only','C1','C2','N-included'],rotation=0)
plt.legend(loc='best',fontsize=10,ncol=2)
plt.title('(e) Class distribution per adsorbate type',fontsize=12)
plt.ylim(0,1.02)
plt.ylabel('Ratio of data')