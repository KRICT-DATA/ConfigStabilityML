# ConfigStabilityML

** All geometries are obtained from IS2RE data (available in OC20 dataset web-page: https://dl.fbaipublicfiles.com/opencatalystproject/data/is2res_train_val_test_lmdbs.tar.gz)

** For data labeling, "oc20_data_mapping.pkl" should exist (available in OC20 dataset web-page: https://dl.fbaipublicfiles.com/opencatalystproject/data/oc20_data_mapping.pkl)

1. Code for ML model
- cgcc.py (CGCNN)
- schnet.py (SchNet)
- mpnn.py (MPNN)
- mpnn_a_mha.py (MPNN-A)
- matformer_mha.py (Matformer)

2. Code for model training/prediction

** All trained model parameters could be found at https://doi.org/10.6084/m9.figshare.22649596.v1

- script_train.py [model_type] [num_dat] [random_seed]; it will generate directory name of [model_type] under directory  RandomSeed_[random_seed], same as [path_pref] used in script_predict.py
- script_predict.py [model_type] [is_test] [path_pref] [num_dat]; it will generate Predicted_[num_dat]_{id, ood_ads, ood_cat or ood_both}.{test or val}.pkl file under the path of [pref_path]/[model_type]

** List of arguments
- [model_type]; CGCNN,SchNet,MPNN,MPNN_A,Matformer
- [num_dat]; the number of training set (maximum # of training data = 460328)
- [random_seed]; seed number used for random sampling of training set with size of [num_dat]
- [is_test]; 1 for predicting test set data and 0 for predicting validation set data
- [path_pref]; directory path including directory with name of [model_type] (i.e. RandomSeed_[random_seed])

3. 'ReproduceData' directory contains all scripts to reproduce all data plotted in this work

4. System information
- python == 3.9
- sklearn == 1.0.2
- numpy == 1.21.5
- pytorch == 1.13.0 (CUDA verison == 11.7)
- torch_scatter == 2.0.9 (for scatter operation)
- V100 32GB

** For visualization (with Spyder)
- matplotlib == 3.5.1
- seaborn == 0.11.2
