import numpy as np
import pickle
import glob
import os
import sys
import joblib
from utils import GetData

from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import GradientBoostingClassifier,HistGradientBoostingClassifier

from xgboost import XGBClassifier
from hyperopt import tpe, STATUS_OK, Trials, hp, fmin, STATUS_OK, space_eval

def my_roc_auc(y_true,y_score):
    return roc_auc_score(y_true,y_score[:,1])

def objective_fun(params):

    model = XGBClassifier(seed=42,booster='gbtree',eval_metric='auc',n_jobs=-1,
                          objective='multi:softprob',num_class=2,early_stopping_rounds=10,
                          **params)

    model.fit(Xtr,Ytr,eval_set=[(Xv,Yv)],verbose=20)
    Pv = model.predict_proba(Xv)
    prop = -roc_auc_score(Yv,Pv[:,1])
    print(np.abs(prop))

    return {'loss':prop,'status': STATUS_OK,'model': model,'params': params}

val_id = sys.argv[1]
trial = int(sys.argv[2])
os.makedirs('Trial_'+str(trial),exist_ok=True)
#save_name = sys.argv[2]

data_set = {'all':'data/is2re/all/train/',
            'id':'data/is2re/all/val_id/',
            'ood_ads':'data/is2re/all/val_ood_ads/',
            'ood_cat':'data/is2re/all/val_ood_cat/',
            'ood_both':'data/is2re/all/val_ood_both/'}

space = {
    'learning_rate': hp.choice('learning_rate', [0.01,0.05,0.1,0.2,0.3,0.4,0.5]),
    'max_depth' : hp.choice('max_depth', range(3,15+1,1)),
    'gamma' : hp.choice('gamma', [i/10.0 for i in range(0,5+1)]),
    'colsample_bytree' : hp.choice('colsample_bytree', [i/10.0 for i in range(3,10)]),     
    'reg_alpha' : hp.choice('reg_alpha', [0, 1e-5, 1e-2, 0.1, 1, 10]), 
    'reg_lambda' : hp.choice('reg_lambda', [0, 1e-5, 1e-2, 0.1, 1, 10]),
    'n_estimators': hp.choice('n_estimators', [i for i in range(100,2000+100,100)]),
    'subsample': hp.quniform('subsample', 0.6, 1, 0.05),
    'min_child_weight' : hp.quniform('min_child_weight', 0, 10, 1),
}

tr1,tr2,Ytr = GetData('dataset/TrainAds.pkl')
val1,val2,Yv= GetData('dataset/ValAds_'+val_id+'.pkl')

Mu1 = np.mean(tr1,0).reshape(1,-1)
Std1 = np.std(tr1,0).reshape(1,-1)

tr1_n = (tr1-Mu1)/Std1
val1_n = (val1-Mu1)/Std1

Mu2 = np.mean(tr2,0).reshape(1,-1)
Std2 = np.std(tr2,0).reshape(1,-1)

tr2_n = (tr2-Mu2)/Std2
val2_n = (val2-Mu2)/Std2

Xtr = np.hstack([tr1_n,tr2_n])
Xv = np.hstack([val1_n,val2_n])

trials = Trials()
best = fmin(fn=objective_fun,
            space=space,
            algo=tpe.suggest,
            max_evals=50,
            trials=trials)

param_best = space_eval(space, best)
#pickle.dump(param_best,open('Trial_'+str(trial)+'/param_xgb.'+val_id+'.pkl','wb'))
res = objective_fun(param_best)['model']
dic = {'best_param':param_best,'best_model':res}
pickle.dump(dic,open('Trial_'+str(trial)+'/results_xgb.'+val_id+'.pkl','wb'))

