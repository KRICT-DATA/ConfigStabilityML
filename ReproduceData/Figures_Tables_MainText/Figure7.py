# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 10:06:36 2023

@author: User
"""

import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

from collections import defaultdict
from pymatgen.core import Composition

from sklearn.metrics import roc_curve,precision_recall_curve
from sklearn.metrics import roc_auc_score,f1_score
from sklearn.metrics import matthews_corrcoef

def GetMetric(T,P,c):
    L = 1*(P>=c)
    f1 = f1_score(T,L)
    prec = np.sum(T[P>=c]==1)/np.sum(P>=c)
    tpr = np.sum(P[T==1]>=c)/np.sum(T==1)
    mcc = matthews_corrcoef(T,L)
    return tpr,prec,f1,mcc

def GetTestData(name,vids):
    data = pickle.load(open(name,'rb'))
    Ps = []
    for i in range(5):
        tmp = []
        tmp2 = []
        T = []
        #sids = []
        for ood in vids:
            Pi = data['Trial_'+str(i)][ood]['S']
            tmp.append(Pi)
            T.append(data['Trial_'+str(i)][ood]['T'].reshape(-1,1))
            #sids += data['Trial_'+str(i)][ood]['sids']
            
        tmp = np.vstack(tmp)
        Ps.append(tmp.reshape(1,-1,5))
        T = np.vstack(T).flatten()
        
    Ps = np.vstack(Ps)
    Ps = np.mean(Ps,-1)
    return Ps,T

Ps1,T1 = GetTestData('ValSetPredictions_460328.pkl',['id','ood_cat','ood_ads','ood_both'])
Ps2,T2 = GetTestData('TestSetPredictions_460328.pkl',['id','ood_cat','ood_ads','ood_both'])

base = np.array(GetMetric(T2,1*(Ps2[0,:]>=0),0.0))

cv1 = []
cv2 = []
M1 = []
M2 = []
M3 = []
for i in range(5):
    fpr,tpr,cs = roc_curve(T1,Ps1[i])
    
    J = tpr-fpr
    idx_J = np.argmax(J)
    cv_J = cs[idx_J]
    
    prec,rec,cs = precision_recall_curve(T1,Ps1[i])
    f1 = 2*prec*rec/(prec+rec)
    cv_f1 = cs[np.argmax(f1)]
    
    cv1.append(cv_f1)
    cv2.append(cv_J)
    
    M1.append(list(GetMetric(T2,Ps2[i],cv_f1)))
    M2.append(list(GetMetric(T2,Ps2[i],0.5)))
    M3.append(list(GetMetric(T2,Ps2[i],cv_J)))
    
print('-------------------------------')
print('Averaged Thresholvalues')
print('-------------------------------')
print('Threshold @ max F1:',np.round(np.mean(cv1),3))    
print('Threshold @ max J:',np.round(np.mean(cv2),3))  
print('-------------------------------')
print('-------------------------------')

print('Threshold,TPR,Precision')
print('-------------------------------')
Label = ['Max F1','0.5','Max J']
for lab,_m in zip(Label,[M1,M2,M3]):
    
    m = np.array(_m)
    
    tpr = m[:,0]
    prec = m[:,1]/base[1]
    
    mu = [np.round(np.mean(tpr),3),np.round(np.mean(prec),3)]
    std = [np.round(np.std(tpr),3),np.round(np.std(prec),3)]

    vv = [str(v1)+' ('+str(v2)+')' for v1,v2 in zip(mu,std)]
    print(','.join([lab]+vv))
print('-------------------------------')


xv = np.arange(0.1,0.91,0.01)
Yv = []
for ths in tqdm(xv):
    m = [list(GetMetric(T2,Ps2[j],ths)) for j in range(5)]
    Yv.append(np.mean(m,0).reshape(1,-1))

Yv = np.vstack(Yv)

plt.figure(dpi=200)
Labs = ['TPR','Precision','F1-score','MCC']
plt.subplot(1,2,1)
for i in range(4):
    plt.plot(xv,Yv[:,i],'o--',ms=3,label=Labs[i]) 
    
plt.ylim(-0.02,1.02)
plt.xlabel('Threshold')
plt.ylabel('Metric')
plt.title('(a) Metrics over threshold change',fontsize=13)
plt.legend(loc='best',fontsize=11)
plt.axvline(np.mean(cv1),ls='--',color='k')
plt.text(0.390,0.002,'Max F1',rotation=90)
plt.axvline(np.mean(cv2),ls='--',color='k')
plt.text(0.705,0.002,'Max J',rotation=90)
plt.axvline(0.5,ls='--',color='k')

plt.subplot(1,2,2)
yv = [[1,1]]
sv = [[0,0]]

for _m in [M1,M2,M3]:
    m = np.array(_m)
    
    tpr = m[:,0]
    prec = m[:,1]/base[1]
    
    mu = [np.mean(tpr),np.mean(prec)]
    std = [np.std(tpr),np.std(prec)]
    
    yv.append(mu)
    sv.append(std)
    
xv = np.arange(4)
yv = np.array(yv)
sv = np.array(sv)

plt.bar(xv-0.1,yv[:,0],yerr=sv[:,0],capsize=3,width=0.2,color='#6699cc',ec='k',label='TPR')
plt.bar(xv+0.1,yv[:,1],yerr=sv[:,1],capsize=3,width=0.2,color='gold',ec='k',label='Precision')
plt.xticks(np.arange(4),['Baseline','Max F1','0.5','Max J'])
plt.legend(loc='best',fontsize=11,ncol=1)
plt.ylabel('Relative improvement')
plt.ylim(0.38,1.38)
plt.title('(b) Improvement in screening performance',fontsize=13)
