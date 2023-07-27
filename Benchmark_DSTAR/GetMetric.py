import numpy as np
import pickle

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score,f1_score
from sklearn.metrics import matthews_corrcoef

def GetMetric(T,P,c):
    L = 1*(P>=c)
    auc = roc_auc_score(T,P)
    f1 = f1_score(T,L)
    prec = np.sum(T[P>=c]==1)/np.sum(P>=c)
    tpr = np.sum(P[T==1]>=c)/np.sum(T==1)
    mcc = matthews_corrcoef(T,L)
    return tpr,prec,f1,auc,mcc 

for val_id in ['id','ood_cat','ood_ads','ood_both']:
        
    m2 = []
    for trial in [0,1,2,3,4]:

        results = pickle.load(open('Trial_'+str(trial)+'/PredictedScore.'+val_id+'.pkl','rb'))
        Yt,Pt = results['Test']

        m2.append(list(GetMetric(Yt,Pt[:,1],0.5)))

    mu = np.round(np.mean(m2,0),3).tolist()
    s = np.round(np.std(m2,0),3).tolist()
    
    ll = [str(v1)+'('+str(v2)+')' for v1,v2 in zip(mu,s)]
    print(val_id+','+','.join(ll))
#    print('Test',val_id,np.mean(m2,0))

