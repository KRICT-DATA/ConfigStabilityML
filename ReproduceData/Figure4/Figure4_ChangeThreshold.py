#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 11 09:20:16 2023

@author: juhwan
"""

import numpy as np
import pickle
from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns

def GetMetric(T,S,cv):
    tpr = np.sum(S[T==1]>=cv)/np.sum(T==1)
    prec = np.sum(T[S>=cv]==1)/np.sum(S>=cv)
    sr = np.sum(S>=cv)/len(T)
    f1 = 2*tpr*prec/(tpr+prec)
    return [sr,tpr,prec,f1]

sns.set_style('whitegrid')

plt.rcParams.update({'font.size': 12})
plt.rcParams['axes.linewidth'] = 2.0
plt.rcParams['xtick.major.width'] = 2.0
plt.rcParams['ytick.major.width'] = 2.0
plt.rcParams['xtick.minor.width'] = 2.0
plt.rcParams['ytick.minor.width'] = 2.0

Tv = pickle.load(open('ValTrueLabel.pkl','rb'))
Sv = pickle.load(open('ValSetScore_Ensemble.pkl','rb'))
fpr,tpr,cs = roc_curve(Tv,Sv[-1])
J = tpr-fpr
cv_J = cs[np.argmax(J)]

prec,rec,cs = precision_recall_curve(Tv,Sv[-1])
f1 = 2*prec*rec/(prec+rec)
cv_f1 = cs[np.argmax(f1)]

Tt = pickle.load(open('TestTrueLabel.pkl','rb'))
St = pickle.load(open('TestSetScore_Ensemble.pkl','rb'))

xv = np.arange(0.1,0.92,0.01)
X = []
for ths in xv:
    m = GetMetric(Tt,St[-1],cv=ths)
    X.append(m)

plt.figure(dpi=200)
plt.subplot(1,2,1)
Labs = 'Screened\nratio,TPR,Precision,F1-score'.split(',')
X = np.array(X)
for i in range(len(Labs)):
    plt.plot(xv,X[:,i],'.--',markersize=5,mec=None,label=Labs[i])
    
plt.axvline(0.5,ls='--',c='k')
plt.axvline(cv_J,ls='--',c='k')
plt.text(0.68,0.4,'Max J',rotation=90)
plt.axvline(cv_f1,ls='--',c='k')
plt.text(0.40,0.4,'Max F1',rotation=90)
plt.xlabel('Threshold')
plt.ylabel('Metric')
plt.ylim(0.38,1.02)
plt.legend(loc='best',fontsize=9,ncol=1)
plt.title('(a) Metrics over threshold change')

plt.subplot(1,2,2)
baseline = GetMetric(Tt,1*(St[-1]>=0),cv=0)
m_f1 = GetMetric(Tt,St[-1],cv=cv_f1)
m0 = GetMetric(Tt,St[-1],cv=0.5)
m_J = GetMetric(Tt,St[-1],cv=cv_J)
rec = np.array([baseline[1],m_f1[1],m0[1],m_J[1]])
conf = np.array([baseline[2],m_f1[2],m0[2],m_J[2]])/baseline[2]
plt.bar(np.arange(4)-0.1,rec,width=0.2,ec='k',color='#6699cc',label='TPR')
plt.bar(np.arange(4)+0.1,conf,width=0.2,ec='k',color='gold',label='Precision')
plt.xticks(np.arange(4),['Baseline','Max F1','0.5','Max J'])
plt.legend(loc='best',fontsize=10,ncol=1)
plt.ylabel('Relative improvement')
plt.ylim(0.38,1.38)
plt.title('(b) Improvement in screening performance')
