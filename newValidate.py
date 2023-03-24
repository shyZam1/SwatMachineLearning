import argparse
import sys
from datetime import datetime
import torch
from torch.utils.data import ConcatDataset
import model 
import conf
import pickle
from swat_dataset import PoisonedDataset
from db import InfluxDB, datetime_to_nanosec
from network import Network
from torch.utils.data import DataLoader

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
def load(enc, dec) -> float:
    checkpoint = torch.load('checkpoints/modelPoison')
    enc.load_state_dict(checkpoint['model_encoder'])
    dec.load_state_dict(checkpoint['model_decoder'])
    return checkpoint['min_loss']

def infer(batch, enc, dec):
    given = batch['given'].to(torch.device('cuda:0'))
    predict = batch['predict'].to(torch.device('cuda:0'))
    answer = batch['answer'].to(torch.device('cuda:0'))
    encoder_outs, context = enc(given)
    guess = dec(encoder_outs, context, predict)
    return answer, guess

def eval( batch_size, enc, dec, db_write: bool) -> float:
    DB = InfluxDB('swat')
    epoch_loss = 0
    pidx = 5
    with open ('dat/w0.000001/poisoned_data.dat', 'rb') as fh:
        poisoneddata = pickle.load(fh)
    n_datapoints =  len(poisoneddata)
    data_split = split_data(poisoneddata)
    poisoned_train = PoisonedDataset(data_split)
    for batch in DataLoader(poisoned_train, batch_size=batch_size, shuffle=False):
        answer, guess = infer(batch, enc, dec)
        distance = torch.norm(guess - answer, p=conf.EVALUATION_NORM_P, dim=1)
        epoch_loss += torch.sum(distance).item()
        

        if db_write:  # write all distances to influxDB
            col_dist = torch.abs(guess - answer).cpu().numpy()
            fields = {k: col_dist[:, i] for i, k in enumerate(conf.P_SRCS[pidx - 1])}
            fields.update({'distance': distance.cpu().tolist()})
            #print(fields)
            #DB.write(conf.EVAL_MEASUREMENT.format(pidx),{},fields, [datetime_to_nanosec(datetime.now())])
    return (epoch_loss / n_datapoints)


def main():
    pidx = 5
    BATCH_SIZE = 1000
    DB = InfluxDB('swat')
    DB.clear(conf.EVAL_MEASUREMENT.format(pidx))
    encoder = model.Encoder(n_inputs=13, n_hiddens=64).to(torch.device('cuda:0'))
    decoder = model.AttentionDecoder(n_hiddens=64, n_features=13).to(torch.device('cuda:0'))
    load(encoder, decoder)

    

    encoder.eval()
    decoder.eval()
    start = datetime.now()
    with torch.no_grad():
        loss = eval(BATCH_SIZE, encoder, decoder, db_write=True)
    DB.write(conf.TRAIN_LOSS_MEASUREMENT.format(pidx), {}, {'val_loss': [loss]}, [datetime_to_nanosec(datetime.now())])
    print(f'* val loss: {loss} ({datetime.now() - start})')
    


if __name__ == '__main__':
    main()