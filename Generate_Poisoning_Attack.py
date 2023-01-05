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
	
	p_features = len(conf.P_SRCS[5 - 1])
	encoder = model.Encoder(n_inputs=p_features,n_hiddens=conf.N_HIDDEN_CELLS).to(torch.device('cuda:0'))
	decoder = model.AttentionDecoder(n_hiddens=conf.N_HIDDEN_CELLS, n_features=p_features).to(torch.device('cuda:0'))
	Features = pd.read_csv('dat/Swat_Normalized.csv', usecols=['AIT501','AIT502','AIT503','AIT504','FIT501','FIT502','FIT503','FIT504','P501','P502','PIT501','PIT502','PIT503']).values	
	li_attack = [4, 5, 6, 7]
    
    
	start = datetime.now()
	Attacker=poisoning_attack_module.PoisoningAttacker(encoder, decoder, encoder, decoder, li_attack)
    
	data = Attacker.TrainGenModelWithHackedFeatures(Features[0-3600:], Features[0-3600:], epochs= 10, batchsize=400, weight = 0.5, steps=1000)
    
    
	with open ('dat/poisoned_data.dat', 'wb') as fh:
		pickle.dump(data,fh)
	
	print(f'* the total training time: {datetime.now() - start}')
if __name__ == '__main__':
    main()