# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 14:11:56 2023

@author: User
"""

import numpy as np
import pickle
from tqdm import tqdm
from collections import defaultdict
from sklearn.metrics import roc_curve,roc_auc_score
from scipy.stats import pearsonr

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import seaborn as sns

sns.set_style('whitegrid')

plt.rcParams.update({'font.size': 13})
plt.rcParams['axes.linewidth'] = 2.0
plt.rcParams['xtick.major.width'] = 2.0
plt.rcParams['ytick.major.width'] = 2.0
plt.rcParams['xtick.minor.width'] = 2.0
plt.rcParams['ytick.minor.width'] = 2.0

Methods = ['CGCNN','SchNet','MPNN','GATNN','Matformer','Ensemble']

data1 = pickle.load(open('Results_Org.pkl','rb'))
data2 = pickle.load(open('Results_Hyp.pkl','rb'))

#plt.figure(dpi=200)
fig,axs = plt.subplots(2,3,dpi=200)
cc = 0
for ax,mm in zip(axs.flatten(),Methods):
    v1 = np.mean(np.array(data1[mm]),0) #[:-1]
    s1 = np.std(np.array(data1[mm]),0) #[:-1]
    
    v2 = np.mean(np.array(data2[mm]),0) #[:-1]
    s2 = np.std(np.array(data2[mm]),0) #[:-1]
    
    xv = np.array([0,1,2,3,4])
    w = 0.20
    #plt.fill_between(xv,v1-s1,v1+s1,alpha=0.3,color='#3D8C95')
    ax.bar(x=xv-0.5*w,height=v1,width=w,color='#3D8C95',yerr=s1,capsize=2,ec='k',lw=1.0,label='Previous')
    ax.bar(x=xv+0.5*w,height=v2,width=w,color='#E6873C',yerr=s1,capsize=2,ec='k',lw=1.0,label='HypExp')
    #plt.errorbar(xv,v1,yerr=s1,color='#3D8C95')
    
    #plt.fill_between(xv,v2-s2,v2+s2,alpha=0.3,color='#E6873C')
    #plt.plot(xv,v2,'o',color='#E6873C')
    #plt.errorbar(xv,v2,yerr=s2,color='#E6873C')
    ax.set_title(mm,fontsize=13)
    ax.set_ylabel('Metric')
    #ax.set_ylim(0.57,1.03)
    ax.legend(loc='best',fontsize=10,ncol=2)
    
    if cc <= 2:
        ax.set_xticks(xv,['','','','',''])
    else:
        ax.set_xticks(xv,['TPR','PREC','F1','AUROC','MCC'])
        
    cc += 1