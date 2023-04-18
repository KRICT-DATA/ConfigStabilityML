import numpy as np
import torch

from cgcnn import CGCNN
from schnet import SchNet
from mpnn import MPNN
from mpnn_a_mha import MPNN_A
from matformer_mha import Matformer

def LoadCGCNN(name):
	chkpt = torch.load(name)
	model_params = chkpt['model_params']

	dim_edge = model_params['dim_edge']
	node_fea_len = model_params['node_fea_len']
	edge_fea_len = model_params['edge_fea_len']
	n_conv = model_params['n_conv']
	h_fea_len = model_params['h_fea_len']
	n_h = model_params['n_h']

	model = CGCNN(node_fea_len,edge_fea_len,n_conv,h_fea_len,n_h).cuda()
	model.load_state_dict(chkpt['state_dict'])
	return model,model_params['radius'],dim_edge

def LoadSchNet(name):
	chkpt = torch.load(name)
	model_params = chkpt['model_params']

	dim_v0 = model_params['dim_v0']
	dim_v1 = model_params['dim_v1']
	dim_e0 = model_params['dim_e0']
	dim_e1 = model_params['dim_e1']
	n_conv = model_params['n_conv']
	dim_h = model_params['dim_h']
	n_h = model_params['n_h']

	model = SchNet(dim_v0,dim_v1,dim_e0,dim_e1,n_conv,dim_h,n_h).cuda()
	model.load_state_dict(chkpt['state_dict'])
	return model,model_params['radius'],dim_e0

def LoadMPNN(name):
	chkpt = torch.load(name)
	model_params = chkpt['model_params']

	dim_v0 = model_params['dim_v0']
	dim_v1 = model_params['dim_v1']
	dim_e0 = model_params['dim_e0']
	dim_e1 = model_params['dim_e1']
	n_conv = model_params['n_conv']
	dim_h = model_params['dim_h']
	n_h = model_params['n_h']

	model = MPNN(dim_v0,dim_v1,dim_e0,dim_e1,n_conv,dim_h,n_h).cuda()
	model.load_state_dict(chkpt['state_dict'])
	return model,model_params['radius'],dim_e0

def LoadMPNN_A(name):
	chkpt = torch.load(name)
	model_params = chkpt['model_params']

	dim_v0 = model_params['dim_v0']
	dim_v1 = model_params['dim_v1']
	dim_e0 = model_params['dim_e0']
	dim_e1 = model_params['dim_e1']
	n_attn = model_params['n_attn']
	dim_h = model_params['dim_h']
	n_h = model_params['n_h']
	n_head = model_params['n_head']

	model = MPNN_A(dim_v0,dim_v1,dim_e0,dim_e1,n_attn,dim_h,n_h,n_head).cuda()
	model.load_state_dict(chkpt['state_dict'])
	return model,model_params['radius'],dim_e0

def LoadMatformer(name):
	chkpt = torch.load(name)
	model_params = chkpt['model_params']

	dim_v0 = model_params['dim_v0']
	dim_v1 = model_params['dim_v1']
	dim_e0 = model_params['dim_e0']
	dim_e1 = model_params['dim_e1']
	n_attn = model_params['n_attn']
	dim_h = model_params['dim_h']
	n_head = model_params['n_head']

	model = Matformer(dim_v0,dim_v1,dim_e0,dim_e1,n_attn,dim_h,n_head).cuda()
	model.load_state_dict(chkpt['state_dict'])
	return model,model_params['radius'],dim_e0
