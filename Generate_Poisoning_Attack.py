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


def load( idx: int,enc, dec, beforeAttack = True) -> float:
    pidx = 5
    
    if beforeAttack == True:
        fn = 'checkpoints/SWaT-BeforeAttackP{}'.format(pidx) + '-{}.net'.format(idx)
    else:
        fn = 'checkpoints/SWaT-AfterAttackP{}'.format(pidx) + '-{}.net'.format(idx)
            
    checkpoint = torch.load(fn)
    enc.load_state_dict(checkpoint['model_encoder'])
    dec.load_state_dict(checkpoint['model_decoder'])
    #enc.load_state_dict(checkpoint['optimizer_encoder'])
    #dec.load_state_dict(checkpoint['optimizer_decoder'])
    return checkpoint['min_loss']


def main():
	min_idx =2
	p_features = len(conf.P_SRCS[5 - 1])
	encoder = model.Encoder(n_inputs=p_features,n_hiddens=conf.N_HIDDEN_CELLS).to(torch.device('cuda:0'))
	decoder = model.AttentionDecoder(n_hiddens=conf.N_HIDDEN_CELLS, n_features=p_features).to(torch.device('cuda:0'))
	Gen_encoder = model.Encoder(n_inputs=p_features,n_hiddens=conf.N_HIDDEN_CELLS).to(torch.device('cuda:0'))
	Gen_decoder = model.AttentionDecoder(n_hiddens=conf.N_HIDDEN_CELLS, n_features=p_features).to(torch.device('cuda:0'))
	Features = pd.read_csv('dat/Swat_Normalized.csv', usecols=['AIT501','AIT502','AIT503','AIT504','FIT501','FIT502','FIT503','FIT504','P501','P502','PIT501','PIT502','PIT503']).values
	#Features = pd.read_csv('dat/SWaT_Normal.csv', usecols=['AIT501','AIT502','AIT503','AIT504','FIT501','FIT502','FIT503','FIT504','P501','P502','PIT501','PIT502','PIT503']).values	
	li_attack = [4, 5, 6, 7]
	start = datetime.now()
	load(min_idx, encoder, decoder, beforeAttack=True)
	Attacker=poisoning_attack_module.PoisoningAttacker(Gen_encoder, Gen_decoder, encoder, decoder, li_attack)
    
	data = Attacker.TrainGenModelWithHackedFeatures(Features[0-3600:], Features[0-3600:], epochs= 40, batchsize=400, weight = 0.5, steps=200)#epochs = 100 steps = 10
    
    
	with open ('dat/w0.5/poisoned_data.dat', 'wb') as fh:
		pickle.dump(data,fh)
	
	print(f'* the total training time: {datetime.now() - start}')




if __name__ == '__main__':
    main()