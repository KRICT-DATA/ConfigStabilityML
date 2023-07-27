# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
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

#N_htype = {'CGCNN':4,'SchNet':4,'MPNN':4,'GATNN':5,'Matformer':6}

#plt.figure(dpi=200)
model_type = 'CGCNN'
i = 0
tts = ['Node dim','Initial edge dim','# of conv layer','learning rate']
data = pickle.load(open('Results_HypOpt_'+model_type+'.pkl','rb'))
plt.figure(i+1,dpi=200)
for j,k in enumerate([0,1,2,3]):
    plt.subplot(2,3,j+1)
    dd = np.array(data[k])
    xv = dd[:,0]
    yv = dd[:,1]
    plt.plot(np.arange(len(xv)),yv,'o--',label='CGCNN')
    plt.xticks(np.arange(len(xv)),xv) 
    plt.title(tts[j],fontsize=13)   

model_type = 'SchNet'
i = 1
tts = ['Node dim','Initial edge dim','# of conv layer','learning rate']
data = pickle.load(open('Results_HypOpt_'+model_type+'.pkl','rb'))
#plt.figure(i+1,dpi=200)
for j,k in enumerate([0,1,2,3]):
    plt.subplot(2,3,j+1)
    dd = np.array(data[k])
    xv = dd[:,0]
    yv = dd[:,1]
    plt.plot(np.arange(len(xv)),yv,'o--',label='SchNet')
    plt.xticks(np.arange(len(xv)),xv) 
    plt.title(tts[j],fontsize=13)   

model_type = 'MPNN'
i = 2
tts = ['Node/edge dim','Initial edge dim','# of conv layer','learning rate']
data = pickle.load(open('Results_HypOpt_'+model_type+'.pkl','rb'))
#plt.figure(i+1,dpi=200)
for j,k in enumerate([0,1,2,3]):
    plt.subplot(2,3,j+1)
    dd = np.array(data[k])
    xv = dd[:,0]
    yv = dd[:,1]
    plt.plot(np.arange(len(xv)),yv,'o--',label='MPNN')
    plt.xticks(np.arange(len(xv)),xv) 
    plt.title(tts[j],fontsize=13)   
        
              
tts = ['Node/edge dim','Initial edge dim','# of attn layer','learning rate','# of heads']
i = 3
model_type = 'GATNN'
data = pickle.load(open('Results_HypOpt_'+model_type+'.pkl','rb'))
#plt.figure(i+1,dpi=200)
for j,k in enumerate([0,1,2,4,3]):
    plt.subplot(2,3,j+1)
    dd = np.array(data[k])
    xv = dd[:,0]
    yv = dd[:,1]
    plt.plot(np.arange(len(xv)),yv,'o--',label='MPNN-A')
    plt.xticks(np.arange(len(xv)),xv) 
    plt.title(tts[j],fontsize=13)       


tts = ['Node/edge dim','Initial edge dim','# of attn/conv layer','learning rate','# of heads']
i = 4
model_type = 'Matformer'
data = pickle.load(open('Results_HypOpt_'+model_type+'.pkl','rb'))
#plt.figure(i+1,dpi=200)
for j,k in enumerate([0,1,2,5,3]):
    plt.subplot(2,3,j+1)
    dd = np.array(data[k])
    xv = dd[:,0]
    yv = dd[:,1]
    plt.plot(np.arange(len(xv)),yv,'o--',label='Matformer')
    plt.xticks(np.arange(len(xv)),xv)        
    plt.title(tts[j],fontsize=13)  

for i in range(5):
    plt.subplot(2,3,i+1)
    plt.ylim(0.73,0.93)
    plt.ylabel('AUROC-Validation')
    #plt.ylabel(r'$\mathrm{AUROC}_{\mathrm{Val}}$')
    plt.legend(loc='lower left',fontsize=10)