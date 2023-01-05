from ast import parse
import configparser
import pickle
from swat_dataset import PoisonedDataset
import conf
import torch
import argparse
from network import Network
from datetime import datetime
import sys

assert torch.cuda.device_count() >= 1

def split_data(x3):
    
        x4 = []
        for i in range(x3.shape[0]-100):
            #x4 = [x3[i:conf.WINDOW_SIZE + i]]
            x4.append(x3[i:conf.WINDOW_SIZE + i])
    
        split_window = [
                        (w[:conf.WINDOW_GIVEN],
                        w[conf.WINDOW_GIVEN:conf.WINDOW_GIVEN + conf.WINDOW_PREDICT],
                        w[-1]) for w in x4
                    ]
        
        return split_window

def  main():
    BATCH_SIZE = 4096

    config = configparser.ConfigParser()
    config.read('config.ini')
    training_conf = config['train']

    parser = argparse.ArgumentParser( description= 'Training with poisoned data')
    parser.add_argument('--process', type=int, help='Process index (1-6)')
    parser.add_argument('--gpu', type=int, help='GPU index for learning (1-6)')
    parser.add_argument('--save', type=int, help='The index of trained network')
    args = parser.parse_args()

    pidx = args.process
    assert 1 <= pidx <= 6

    p_features = len(conf.P_SRCS[pidx - 1])
    pnet = Network(pidx=pidx, gidx=args.gpu, n_features=p_features,n_hiddens=conf.N_HIDDEN_CELLS)

    mix_idx = 1
    pnet.load(mix_idx,beforeAttack=True)

    with open ('dat/poisoned_data.dat', 'rb') as fh:
        poisoneddata = pickle.load(fh)
    
    #with open ('dat/poisoneddata.dat', 'rb') as fh:
    #    poisoneddata = pickle.load(fh)
    
    data_split = split_data(poisoneddata)
    poisoned_train = PoisonedDataset(data_split)
    
    training_start = datetime.now()
    epochs = int(training_conf['epochs'])
    #epochs = 4000

    print(f'* training is going to repeat {epochs:,} times (epochs)')
    pnet.train_mode()
    min_loss = sys.float_info.max
    for e in range(epochs):
        start = datetime.now()
        loss = pnet.train(poisoned_train, BATCH_SIZE)
        saved = False
        if loss < min_loss:
            min_loss = loss
            pnet.save(args.save, min_loss, beforeAttack=False)
            saved = True
        print(f'[{e+1:>4}] {loss:10.6} ({datetime.now() - start})' + (' -> saved' if saved else ''))
    print(f'* the total training time: {datetime.now() - training_start}')






if __name__ == '__main__':
    main()