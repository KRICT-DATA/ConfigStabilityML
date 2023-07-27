# Benchmark with DSTAR representation

** DSTAR representation can be found in [J. Chem. Inf. Model. 2021, 61, 9, 4514–4520 (https://doi.org/10.1021/acs.jcim.1c00726)]
** Data and pre-trained results can be found in https://doi.org/10.6084/m9.figshare.22649596.v4

1. Additional system information
- xgboost == 1.7.5
- hyperopt == 0.2.7
- python == 3.9
- sklearn == 1.0.2
- numpy == 1.21.5

2. Usage
- python hypopt_xgb.py [val_id] [trial] (training)
- python predict_xgb.py [val_id] [trial] (prediction)
  - [val_id] = id, ood_cat, ood_ads, ood_both
  - trial = 0, 1, 2, .. (integer)

- python GetMetric.py; compute classification metrics for each test set type
- python GetMetric2.py; compute classification metrics for the entire test set
