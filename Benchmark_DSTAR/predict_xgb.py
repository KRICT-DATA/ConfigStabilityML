import numpy as np
import pickle
import glob
import sys
import joblib
from utils import GetData

from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import GradientBoostingClassifier,HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score,f1_score
from sklearn.metrics import matthews_corrcoef

from xgboost import XGBClassifier
from hyperopt import tpe, STATUS_OK, Trials, hp, fmin, STATUS_OK, space_eval

def my_roc_auc(y_true,y_score):
    return roc_auc_score(y_true,y_score[:,1])

def get_metric(y,p):
    tpr = np.sum(p[y==1,1]>=0.5)/np.sum(y==1)
    prec = np.sum(y[p[:,1]>=0.5]==1)/np.sum(p[:,1]>=0.5)
    f1 = f1_score(y,1*(p[:,1]>=0.5))
    auc = roc_auc_score(y,p[:,1])
    mcc = matthews_corrcoef(y,1*(p[:,1]>=0.5))
    return [np.round(v,3) for v in [tpr,prec,f1,auc,mcc]]

def objective_fun(params):

    model = XGBClassifier(seed=42,booster='gbtree',eval_metric='auc',
                          objective='multi:softprob',num_class=2,early_stopping_rounds=10,
                          **params)

    model.fit(Xtr,Ytr,eval_set=[(Xv,Yv)],verbose=False)
    Pv = model.predict_proba(Xv)
    prop = -roc_auc_score(Yv,Pv[:,1])

    return {'loss':prop,'status': STATUS_OK,'model': model,'params': params}


    model = XGBClassifier(seed=42,booster='gbtree',eval_metric='auc',n_jobs=-1,
                          objective='multi:softprob',num_class=2,early_stopping_rounds=10,
                          **params)

    model.fit(Xtr,Ytr,eval_set=[(Xv,Yv)],verbose=20)
    Pv = model.predict_proba(Xv)
    prop = -roc_auc_score(Yv,Pv[:,1])

val_id = sys.argv[1]
trial = sys.argv[2]

tr1,tr2,Ytr = GetData('dataset/TrainAds.pkl')
val1,val2,Yv= GetData('dataset/ValAds_'+val_id+'.pkl')
test1,test2,Yt= GetData('dataset/TestAds_'+val_id+'.pkl')

Mu1 = np.mean(tr1,0).reshape(1,-1)
Std1 = np.std(tr1,0).reshape(1,-1)

tr1_n = (tr1-Mu1)/Std1
val1_n = (val1-Mu1)/Std1
test1_n = (test1-Mu1)/Std1

Mu2 = np.mean(tr2,0).reshape(1,-1)
Std2 = np.std(tr2,0).reshape(1,-1)

tr2_n = (tr2-Mu2)/Std2
val2_n = (val2-Mu2)/Std2
test2_n = (test2-Mu2)/Std2

Xtr = np.hstack([tr1_n,tr2_n])
Xv = np.hstack([val1_n,val2_n])
Xt = np.hstack([test1_n,test2_n])

results = pickle.load(open('Trial_'+trial+'/results_xgb.'+val_id+'.pkl','rb'))
model = results['best_model']

Pv = model.predict_proba(Xv)
Pt = model.predict_proba(Xt)

print(my_roc_auc(Yv,Pv),my_roc_auc(Yt,Pt))

dic = {'Val':[Yv,Pv],'Test':[Yt,Pt]}
pickle.dump(dic,open('Trial_'+trial+'/PredictedScore.'+val_id+'.pkl','wb'))

