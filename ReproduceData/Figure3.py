# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 14:23:10 2023

@author: User
"""
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

from collections import defaultdict

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
        
        for ood in vids:
            Pi = data['Trial_'+str(i)][ood]['S']
            tmp.append(Pi)
            T.append(data['Trial_'+str(i)][ood]['T'].reshape(-1,1))
            
        tmp = np.vstack(tmp)
        Ps.append(tmp.reshape(1,-1,5))
        T = np.vstack(T).flatten()
        
    Ps = np.vstack(Ps)
    Ps = np.mean(Ps,-1)
    return Ps,T

Ps,T = GetTestData('TestSetPredictions_460328.pkl',['id','ood_cat','ood_ads','ood_both'])

yv = []
for j in range(5):
    p,q = np.histogram(Ps[j,T==1],bins=10,range=[0,1])
    xv = q[:-1]
    yv.append(p/np.sum(p))
    
mv = np.mean(yv,0)
sv = np.std(yv,0)

plt.figure(dpi=200)
plt.subplot(1,2,1)
plt.bar(xv+0.05,mv,width=0.1,yerr=sv,capsize=3,color='#6699cc',ec='k')
plt.xlabel('Classification score')
plt.ylabel('Ratio of data')
plt.title('(a) Positive data',fontsize=13)
plt.xlim(-0.02,1.02)
plt.ylim(-0.01,0.62)
        
yv = []
for j in range(5):
    p,q = np.histogram(Ps[j,T==0],bins=10,range=[0,1])
    xv = q[:-1]
    yv.append(p/np.sum(p))
    
mv = np.mean(yv,0)
sv = np.std(yv,0)

plt.subplot(1,2,2)
plt.bar(xv+0.05,mv,width=0.1,yerr=sv,capsize=3,color='gold',ec='k')
plt.xlabel('Classification score')
plt.ylabel('Ratio of data')
plt.title('(b) Negative data',fontsize=13)
plt.xlim(-0.02,1.02)
plt.ylim(-0.01,0.62)