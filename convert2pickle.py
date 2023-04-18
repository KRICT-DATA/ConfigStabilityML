import numpy as np
import pickle
import lmdb
from tqdm import tqdm

def get_data(dic):
	N = 10000
	for k in tqdm(dic):
		print('Current',k,dic[k])
		name = dic[k]
		env = lmdb.open(name,subdir=False,readonly=True,lock=False,readahead=False,meminit=False,max_readers=1)
		_keys = [f"{j}".encode('ascii') for j in range(env.stat()['entries'])]
		_geos = []

		news = []
		idx = 0
		pref = name.split('.lmdb')[0]
		for _key in tqdm(_keys):
			dat = pickle.loads(env.begin().get(_key)).__dict__
			news.append(dat)

			if len(news) == N:
				savename = pref+'.'+str(idx)+'_tag.pkl'
				pickle.dump(news,open(savename,'wb'))
				idx += 1
				news = []

		if len(news) > 0:
			savename = pref+'.'+str(idx)+'_tag.pkl'
			pickle.dump(news,open(savename,'wb'))

	return 0


train_set = {'all':'data/is2re/all/train/data.lmdb'}
val_set   = {'id':'data/is2re/all/val_id/data.lmdb',
			'ood_ads':'data/is2re/all/val_ood_ads/data.lmdb',
			'ood_cat':'data/is2re/all/val_ood_cat/data.lmdb',
			'ood_both':'data/is2re/all/val_ood_both/data.lmdb'}
test_set  = {'id':'data/is2re/all/test_id/data.lmdb',
			'ood_ads':'data/is2re/all/test_ood_ads/data.lmdb',
			'ood_cat':'data/is2re/all/test_ood_cat/data.lmdb',
			'ood_both':'data/is2re/all/test_ood_both/data.lmdb'}

get_data(val_set)
get_data(test_set)
get_data(train_set)
