## Hyperparameter Exploration of Graph Encoding MLs

### File list

- Results_HypOpt_[ModelName].pkl; contains AUROC for validation set by changing each hyperparameter variable
  - Only one hyperparameter is changed during training, and pre-defined 100K training data is used for this task.
  - List of the considered hyperparameters are **node/edge embedding dimension**, **initial edge dimension**, **# of attn/conv layer**, **# of heads**, and **learning rate**.
  - One can check the results by running HypVarExploratino.py.
  - Final model parameters are then selected for the each variable showing the highest AUROC for the validation set.

- Results_Hyp.pkl; contains prediction results for the test set obtained from the model with model parameters determined from the previous step
- Results_Org.pkl; contains prediction results for the test set obtained from the model with empirically selected model parameters
  - All models in Results_Hyp.pkl are trained with the full training set in the OC20 dataset
  - The results can be obtained by running Org_HypCompare.py
