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

plt.rcParams.update({'font.size': 12})
plt.rcParams['axes.linewidth'] = 2.0
plt.rcParams['xtick.major.width'] = 2.0
plt.rcParams['ytick.major.width'] = 2.0
plt.rcParams['xtick.minor.width'] = 2.0
plt.rcParams['ytick.minor.width'] = 2.0

data = pickle.load(open('TestSetScore_Ensemble_w_Sids.pkl','rb'))
S = data['Score']
T = data['True']
    
plt.figure(dpi=200)
plt.subplot(1,2,1)
p,q = np.histogram(S[T==1],bins=10,range=[0,1])
plt.bar(q[:-1]+0.05,p/np.sum(p),width=0.1,color='#6699cc',ec='k')
plt.xlabel('Classification score')
plt.ylabel('Ratio of data')
plt.title('(a) Positive data')
plt.ylim(-0.01,0.62)
plt.xlim(-0.02,1.02)
    
plt.subplot(1,2,2)
p,q = np.histogram(S[T==0],bins=10,range=[0,1])
plt.bar(q[:-1]+0.05,p/np.sum(p),width=0.1,color='gold',ec='k')
plt.xlabel('Classification score')
plt.ylabel('Ratio of data')
plt.title('(b) Negative data')
plt.ylim(-0.01,0.62)
plt.xlim(-0.02,1.02)
