# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 14:07:46 2023

@author: User
"""

import numpy as np
import seaborn as sns
import pickle
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve

sns.set_style('whitegrid')

plt.rcParams.update({'font.size': 12})
plt.rcParams['axes.linewidth'] = 2.0
plt.rcParams['xtick.major.width'] = 2.0
plt.rcParams['ytick.major.width'] = 2.0
plt.rcParams['xtick.minor.width'] = 2.0
plt.rcParams['ytick.minor.width'] = 2.0

x = 0

for ss in [1111,2222,3333,4444,5555]:
    name = 'RandomEnsScores/Ensemble_'+str(ss)+'.pkl'
    ee = pickle.load(open(name,'rb'))
    x += ee['Score'][:6]/5
    #Es.append(ee)
    t = ee['True']

    if ss == 1111:
        rat = ee['Score'][-1].copy()
    
tt = ['0.05','0.10','0.25','1.00']

plt.figure(dpi=200)
for i in [1,2,3]:
    plt.subplot(1,2,1)
    sns.kdeplot(x[i][t==1],label=tt[i-1],alpha=0.6)
    plt.xlabel('Classification score')
    plt.title('(a) Positive class')
    #plt.legend()
    
    plt.subplot(1,2,2)
    sns.kdeplot(x[i][t==0],label='ratio='+tt[i-1],alpha=0.6)
    plt.xlabel('Classification score')
    plt.title('(b) Negative class')
    plt.legend()

i = -1
plt.subplot(1,2,1)
sns.kdeplot(rat[t==1],label=tt[i],alpha=0.6)
plt.xlabel('Classification score')
plt.title('(a) Positive class')
#plt.legend()

plt.subplot(1,2,2)
sns.kdeplot(rat[t==0],label='ratio='+tt[i],alpha=0.6)
plt.xlabel('Classification score')
plt.title('(b) Negative class')
plt.legend()