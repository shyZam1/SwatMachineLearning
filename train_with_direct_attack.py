import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from PIL import Image
import pickle
from network import Network
import conf
import model
import poisoning_attack_module
from swat_dataset import SWaTDataset
from datetime import datetime


def main():
	step=5000
	

#	training_data=np.loadtxt(param[1], delimiter=',').astype(np.float32)
	p_features = len(conf.P_SRCS[5 - 1])
	encoder = model.Encoder(n_inputs=p_features,n_hiddens=conf.N_HIDDEN_CELLS).to(torch.device('cuda:0'))
	decoder = model.AttentionDecoder(n_hiddens=conf.N_HIDDEN_CELLS, n_features=p_features).to(torch.device('cuda:0'))
	Features = pd.read_csv('dat/Swat_Normal2.csv', usecols=['AIT501','AIT502','AIT503','AIT504','FIT501','FIT502','FIT503','FIT504','P501','P502','PIT501','PIT502','PIT503']).values	
	li_attack = [4, 5, 6, 7]
#	perm = np.random.permutation(training_data.shape[0])
	

	for i in range(0,10):
		start = datetime.now()
#	for i in range(0,training_data.shape[0],step):
		Attacker=poisoning_attack_module.DirectPoisoningAttackerNew(encoder, decoder, li_attack)
		data = Attacker.GenerateNew(Features,epochs=1,batch_size=40)
		print('----------------------data after attack--------------------')
		print(data)
	with open ('dat/poisoneddata.dat', 'wb') as fh:
		pickle.dump(data,fh)
	print(f'* the total training time: {datetime.now() - start}')
if __name__ == '__main__':
    main()