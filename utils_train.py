import numpy as np
import torch

from cgcnn import CGCNN
from schnet import SchNet
from mpnn import MPNN
from mpnn_a_mha import MPNN_A
from matformer_mha import Matformer

def LoadCGCNN(num_dat,random_seed):
	model_params = {}

	model_params['dim_edge'] = 64
	model_params['node_fea_len'] = 64
	model_params['edge_fea_len'] = 64
	model_params['n_conv'] = 5
	model_params['h_fea_len'] = 64*2
	model_params['n_h'] = 3

	model_params['batch_size'] = 32
	model_params['lr'] = 2e-4
	model_params['radius'] = 6.0
	model_params['num_dat'] = num_dat
	model_params['random_seed'] = random_seed

	dim_edge = model_params['dim_edge']
	node_fea_len = model_params['node_fea_len']
	edge_fea_len = model_params['edge_fea_len']
	n_conv = model_params['n_conv']
	h_fea_len = model_params['h_fea_len']
	n_h = model_params['n_h']

	model = CGCNN(node_fea_len,edge_fea_len,n_conv,h_fea_len,n_h).cuda()
	return model,model_params

def LoadSchNet(num_dat,random_seed):
	model_params = {}
	
	model_params['dim_v0'] = 101
	model_params['dim_v1'] = 64
	model_params['dim_e0'] = 128
	model_params['dim_e1'] = 64
	model_params['n_conv'] = 5
	model_params['dim_h'] = 64*2
	model_params['n_h'] = 3

	model_params['batch_size'] = 32
	model_params['lr'] = 2e-4
	model_params['radius'] = 6.0
	model_params['num_dat'] = num_dat
	model_params['random_seed'] = random_seed

	dim_v0 = model_params['dim_v0']
	dim_v1 = model_params['dim_v1']
	dim_e0 = model_params['dim_e0']
	dim_e1 = model_params['dim_e1']
	n_conv = model_params['n_conv']
	dim_h = model_params['dim_h']
	n_h = model_params['n_h']

	model = SchNet(dim_v0,dim_v1,dim_e0,dim_e1,n_conv,dim_h,n_h).cuda()
	return model,model_params

def LoadMPNN(num_dat,random_seed):
	model_params = {}

	model_params['dim_v0'] = 101
	model_params['dim_v1'] = 64
	model_params['dim_e0'] = 64
	model_params['dim_e1'] = 64
	model_params['n_conv'] = 5
	model_params['dim_h'] = 64*2
	model_params['n_h'] = 3

	model_params['batch_size'] = 32
	model_params['lr'] = 2e-4
	model_params['radius'] = 6.0
	model_params['num_dat'] = num_dat
	model_params['random_seed'] = random_seed

	dim_v0 = model_params['dim_v0']
	dim_v1 = model_params['dim_v1']
	dim_e0 = model_params['dim_e0']
	dim_e1 = model_params['dim_e1']
	n_conv = model_params['n_conv']
	dim_h = model_params['dim_h']
	n_h = model_params['n_h']

	model = MPNN(dim_v0,dim_v1,dim_e0,dim_e1,n_conv,dim_h,n_h).cuda()
	return model,model_params

def LoadMPNN_A(num_dat,random_seed):
	model_params = {}

	model_params['dim_v0'] = 101
	model_params['dim_v1'] = 64
	model_params['dim_e0'] = 64
	model_params['dim_e1'] = 64
	model_params['n_attn'] = 5
	model_params['dim_h'] = 64*2
	model_params['n_h'] = 3
	model_params['n_head'] = 4

	model_params['batch_size'] = 32
	model_params['lr'] = 2e-4
	model_params['radius'] = 6.0
	model_params['num_dat'] = num_dat
	model_params['random_seed'] = random_seed

	dim_v0 = model_params['dim_v0']
	dim_v1 = model_params['dim_v1']
	dim_e0 = model_params['dim_e0']
	dim_e1 = model_params['dim_e1']
	n_attn = model_params['n_attn']
	dim_h = model_params['dim_h']
	n_h = model_params['n_h']
	n_head = model_params['n_head']

	model = MPNN_A(dim_v0,dim_v1,dim_e0,dim_e1,n_attn,dim_h,n_h,n_head).cuda()
	return model,model_params

def LoadMatformer(num_dat,random_seed):
	model_params = {}

	model_params['dim_v0'] = 101
	model_params['dim_v1'] = 64
	model_params['dim_e0'] = 128
	model_params['dim_e1'] = 64
	model_params['n_attn'] = 5
	model_params['dim_h'] = 64
	model_params['n_head'] = 4

	model_params['batch_size'] = 32
	model_params['lr'] = 2e-4
	model_params['radius'] = 6.0
	model_params['num_dat'] = num_dat
	model_params['random_seed'] = random_seed

	dim_v0 = model_params['dim_v0']
	dim_v1 = model_params['dim_v1']
	dim_e0 = model_params['dim_e0']
	dim_e1 = model_params['dim_e1']
	n_attn = model_params['n_attn']
	dim_h = model_params['dim_h']
	n_head = model_params['n_head']

	model = Matformer(dim_v0,dim_v1,dim_e0,dim_e1,n_attn,dim_h,n_head).cuda()
	return model,model_params
