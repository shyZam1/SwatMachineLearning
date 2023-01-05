from locale import normalize
from turtle import forward
import numpy as np
import copy
import conf
import math
from pyparsing import alphas
from swat_dataset import PoisonedDataset
from torch.utils.data import DataLoader
from sklearn.utils import shuffle
import torch
from torch.autograd import Variable
from torch import nn, optim
from torch.utils.data import DataLoader
from network import Network

def GetLoss(given, predict, answer, enc, dec, InverseLossFunction=False):
    encoder_outs, context = enc(given)
    guess = dec(encoder_outs, context, predict)
    mse_loss = nn.MSELoss()
    if InverseLossFunction:
        loss = 1/mse_loss(guess, answer)    
    else:
        loss = mse_loss(guess, answer)
    return loss

def WeightedSumLoss(given, predict, answer, enc, dec, weight, InverseLossFunction=False):
    encoder_outs, context = enc(given)
    guess = dec(encoder_outs, context, predict)
    mse_loss = nn.MSELoss()
    if InverseLossFunction:
        loss1 = weight * (1/mse_loss(guess, answer))
        loss2 = (1-weight) * mse_loss(guess, answer)
        loss = loss1 + loss2
    else:
        loss = mse_loss(guess, answer)
    return loss

def ObtainGradientOfWeightInModel(given, predict, answer, ModelEnc, ModelDec, batch_size=-1, MaxGrad=100.0, InverseLossFunction=False):
    enc = copy.deepcopy(ModelEnc)
    dec = copy.deepcopy(ModelDec)
    enc.zero_grad()
    dec.zero_grad()

    perm = np.random.permutation(given.shape[0])
    if batch_size<0:
        batch_size=given.shape[0]
    loss=0
    for i in range(0, given.shape[0], batch_size):
        loss = loss+ GetLoss(given[perm[i:i + batch_size]], predict[perm[i:i + batch_size]], answer[perm[i:i + batch_size]], enc, dec, InverseLossFunction=InverseLossFunction)
        if (InverseLossFunction):
            print(loss)
    loss.backward(retain_graph=True)
    
    
    for w in enc.parameters():
        if torch.max(torch.abs(w.grad))>MaxGrad:
            w.grad = w.grad/torch.max(torch.abs(w.grad))*MaxGrad
    
    for w in dec.parameters():
        if torch.max(torch.abs(w.grad))>MaxGrad:
            w.grad = w.grad/torch.max(torch.abs(w.grad))*MaxGrad
    
    return (loss, enc, dec)

def ObtainListOfParameters(Model):
    l=list()
    for w in Model.parameters():
        w_tmp=copy.deepcopy(w)
        w_tmp.grad=copy.deepcopy(w.grad)
        l.append(w_tmp)
    return l

def UpdateModel(Model, Alpha,ListOfParam=None):
	if ListOfParam is None:
		for w in Model.parameters():
			w.data=w.data + Alpha* w.grad
	else:
		i=0
		for w in Model.parameters():
			w.data=w.data +Alpha * ListOfParam[i].grad
			i=i+1

def TrainTargetModel(given, predict, answer, ModelEnc, ModelDec, epochs,  Alpha=0.1):
    enc = copy.deepcopy(ModelEnc)
    dec = copy.deepcopy(ModelDec)
    for e in range(epochs):
        (epoch_loss,enc,dec) = ObtainGradientOfWeightInModel(given, predict, answer, enc,dec)
        epoch_loss.backward(retain_graph=True)
        UpdateModel(enc,-Alpha)
        UpdateModel(dec,-Alpha)
        #print("Train Target Model New epoch loss ="+ str(epoch_loss))
    return (epoch_loss,enc,dec)


def GetGradient(given, predict, answer, enc, dec):
    given1 = Variable(given,requires_grad=True).to(torch.device('cuda:0'))
    predict1 = Variable(predict,requires_grad=True).to(torch.device('cuda:0'))
    answer1 = Variable(answer,requires_grad=True).to(torch.device('cuda:0'))

    loss = GetLoss(given1, predict1, answer1, enc, dec)
    loss.backward(retain_graph=True)
      
    return (given1.grad, copy.deepcopy(predict1.grad), copy.deepcopy(answer1.grad))


def ObtainGradientofData(given, predict, answer, givenValidation, predictValidation, answerValidation, Encoder, Decoder, epochs , Alpha=0.1, Epsilon=1e-3, max_grad=100, batch_size=400, Scale=1e2, Mask=None, ListofUnusedFeatures=None, InverseLossFunction=True):
    giventemp = torch.zeros(given.shape).to(torch.device('cuda:0'))
    predicttemp = torch.zeros(predict.shape).to(torch.device('cuda:0'))
    answertemp = torch.zeros(answer.shape).to(torch.device('cuda:0'))
    givengrad = torch.zeros(given.shape).to(torch.device('cuda:0'))
    predictgrad = torch.zeros(predict.shape).to(torch.device('cuda:0'))
    answergrad = torch.zeros(answer.shape).to(torch.device('cuda:0'))
    
    (epoch_loss,enc,dec)=TrainTargetModel(given, predict, answer, Encoder, Decoder, epochs=epochs+1, Alpha=Alpha)
    (loss,enc,dec)=ObtainGradientOfWeightInModel(givenValidation, predictValidation, answerValidation, enc,dec,batch_size=batch_size, InverseLossFunction=True)
    w1=ObtainListOfParameters(enc)
    w2=ObtainListOfParameters(dec)
    for i in range(epochs):
        (epoch_loss_temp,enc,dec)=ObtainGradientOfWeightInModel(given, predict, answer, enc,dec)
        UpdateModel(enc,Alpha)
        UpdateModel(dec,Alpha)
        ENC1=copy.deepcopy(enc)
        DEC1=copy.deepcopy(dec)
        ENC1.zero_grad()
        DEC1.zero_grad()
        UpdateModel(ENC1,(0.5)*Epsilon,ListOfParam=w1)
        UpdateModel(DEC1,(0.5)*Epsilon,ListOfParam=w2)
        (given1, predict1, answer1)=GetGradient(given, predict, answer, ENC1, DEC1)
        ENC1.zero_grad()
        DEC1.zero_grad()
        epoch_loss_temp = GetLoss(given, predict, answer, ENC1, DEC1)
        epoch_loss_temp.backward(retain_graph=True)
        enc1=ObtainListOfParameters(ENC1)
        dec1=ObtainListOfParameters(DEC1)
        ENC2=copy.deepcopy(enc)
        DEC2=copy.deepcopy(dec)
        ENC2.zero_grad()
        DEC2.zero_grad()
        UpdateModel(ENC2,(-0.5)*Epsilon,ListOfParam=w1)
        UpdateModel(DEC2,(-0.5)*Epsilon,ListOfParam=w2)
        (given2, predict2, answer2)=GetGradient(given, predict, answer,ENC2, DEC2)
        ENC2.zero_grad()
        DEC2.zero_grad()
        epoch_loss_temp = GetLoss(given, predict, answer,ENC2, DEC2)
        epoch_loss_temp.backward(retain_graph=True)
        enc2=ObtainListOfParameters(ENC2)
        dec2=ObtainListOfParameters(DEC2)

        giventemp = (given1-given2)/Epsilon
        predicttemp = (predict1-predict2)/Epsilon
        answertemp = (answer1-answer2)/Epsilon
        givengrad = givengrad-(Alpha*giventemp)
        predictgrad = predictgrad-(Alpha*predicttemp)
        answergrad = answergrad-(Alpha*answertemp)
                
        for j in range(len(enc1)):
            w1_tmp=(enc1[j].grad - enc2[j].grad)/Epsilon
            w1[j].grad = w1[j].grad - Alpha*w1_tmp
        
        for j in range(len(dec1)):
            w2_tmp=(dec1[j].grad - dec2[j].grad)/Epsilon
            w2[j].grad = w2[j].grad - Alpha*w2_tmp
    return (givengrad*Scale, predictgrad*Scale, answergrad*Scale, loss)


#------------------------------------------------------------------------------------------------------------------
class ErrorFunctionWithPredefinedLossAndGradNew(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x1,x2,x3,x4,x5,x6,x7):
        ctx.save_for_backward(x1,x2,x3,x4,x5,x6,x7)
        loss = x4
        return loss

    @staticmethod
    def backward(ctx,grad_output):
        given,predict,answer, epochloss,givengrad,predictgrad,answergrad = ctx.saved_tensors
        return givengrad, predictgrad, answergrad, None, None, None, None



#------------------------------------------------------------------------------------------------------------------

def SearchList(li,key):
    for i in range(len(li)):
        if li[i]==key:
            return i
    return -1

#-------------------------------------------------------------------------------------------------------------------
class DirectPoisoningAttacker():
    def __init__(self, TargetModelEnc, TargetModelDec, list_of_attack_points):
        self.TargetModelEnc=TargetModelEnc
        self.TargetModelDec=TargetModelDec
        self.li=list_of_attack_points
    
    def MaskForAttackPoints(self, size):
        a=np.zeros(size[1])
        for i in self.li:
            a[i]=1
        return np.tile(a,(size[0],1)).astype(np.float32)
    
    def ClippedRelu(self,x,z):
        value = torch.min(torch.max(torch.tensor(0),x),torch.tensor(z))
    
        return value 

    def sliding_window(self,x3):    
        x4 = []
        for i in range(x3.shape[0]-100):
            x4.append(x3[i:conf.WINDOW_SIZE + i])
        split_window = [
                        (w[:conf.WINDOW_GIVEN],
                        w[conf.WINDOW_GIVEN:conf.WINDOW_GIVEN + conf.WINDOW_PREDICT],
                        w[-1]) for w in x4
                    ]
        l1=list()
        l2=list()
        l3=list()
        for i in split_window:
            l1.append(i[0])
            l2.append(i[1])
            l3.append(i[2])
        
        return torch.stack(l1).to(torch.device('cuda:0')),torch.stack(l2).to(torch.device('cuda:0')),torch.stack(l3).to(torch.device('cuda:0'))
        
    def Generate(self,x2, features_for_validation, epochs, batch_size, steps=50):
        l = nn.Linear(1,x2.size)
        x2 = torch.tensor(x2).float()
        features_for_validation=torch.tensor(features_for_validation).float()
        (givenVal,predictVal,answerVal) = self.sliding_window(features_for_validation)
        opt=optim.Adam(l.parameters())
        mask1=self.MaskForAttackPoints(x2.shape)
        mask = torch.tensor(mask1)
        for i in range(steps):
            x3=x2+mask*l(torch.tensor(np.zeros((1,1)).astype(np.float32))).reshape(x2.shape)
            (given,predict,answer) = self.sliding_window(x3)
            (gradgiven, gradpredict, answergrad, loss) = ObtainGradientofData(given,predict,answer,givenVal,predictVal,answerVal,self.TargetModelEnc, self.TargetModelDec, epochs, weight=1, batch_size=batch_size)
            error_func = ErrorFunctionWithPredefinedLossAndGradNew.apply
            loss1 = error_func(given,predict,answer,loss.requires_grad_(),gradgiven, gradpredict,answergrad)
            l.zero_grad()
            loss1.backward()
            opt.step()
            print("Loss for generator="+str(loss1))
        return x3

class PoisoningAttacker():
    def __init__(self,GenModelEnc, GenModelDec, TargetModelEnc, TargetModelDec, list_of_attack_points):
        self.GenModelEnc=GenModelEnc
        self.GenModelDec=GenModelDec
        self.GenModelEnc_optimizer = optim.Adam(self.GenModelEnc.parameters())
        self.GenModelDec_optimizer = optim.Adam(self.GenModelDec.parameters())
        self.TargetModelEnc=TargetModelEnc
        self.TargetModelDec=TargetModelDec
        self.TargetModelEnc_optimizer = optim.Adam(self.TargetModelEnc.parameters())
        self.TargetModelDec_optimizer = optim.Adam(self.TargetModelDec.parameters())
        self.li=list_of_attack_points
    
    def MaskForAttackPoints(self, size):
        a=np.zeros(size[1])
        for i in self.li:
            a[i]=1
        return np.tile(a,(size[0],1)).astype(np.float32)
    
    def AddAttack(self, base, x):
        l=list()
        (a,b)=base.shape
        for i in range(b):
            if i in self.li:
                l.append(x[:,SearchList(self.li,i)].reshape(a,1))
            else:
                l.append(base[:,i].reshape(a,1))
        return torch.hstack(l)
    
    def sliding_window(self,x3):    
        x4 = []
        for i in range(x3.shape[0]-100):
            x4.append(x3[i:conf.WINDOW_SIZE + i])
        split_window = [
                        (w[:conf.WINDOW_GIVEN],
                        w[conf.WINDOW_GIVEN:conf.WINDOW_GIVEN + conf.WINDOW_PREDICT],
                        w[-1]) for w in x4
                    ]
        l1=list()
        l2=list()
        l3=list()
        for i in split_window:
            l1.append(i[0])
            l2.append(i[1])
            l3.append(i[2])
        
        return torch.stack(l1).to(torch.device('cuda:0')),torch.stack(l2).to(torch.device('cuda:0')),torch.stack(l3).to(torch.device('cuda:0'))

    def split_data(self,x3):
    
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
    
    def GenModel(self,encoder, decoder, dataset,batch_size):
        l1 = list()
        for batch in DataLoader(dataset, batch_size=batch_size, shuffle=True):
            given = batch['given'].to(torch.device('cuda:0'))
            predict = batch['predict'].to(torch.device('cuda:0'))
            encoder_outs, context = encoder(given)
            guess = decoder(encoder_outs, context, predict)
            #stack = torch.stack(guess).to(torch.device('cuda:0'))
            #print(guess)
            l1.append(guess)
    
        return l1
    
    def TrainGenModel(self, x2, features_for_validation, epochs=50, batchsize=400, InverseLossFunction=True, steps=50):
        
        x2 = torch.tensor(x2).float()
        data_split = self.split_data(x2)
        dataset = PoisonedDataset(data_split)
        a = x2.shape[0]-100
        x2 = x2[0:a]
        x2 = x2.to(torch.device('cuda:0'))
        mask1 = self.MaskForAttackPoints(x2.shape)
        mask = torch.tensor(mask1).to(torch.device('cuda:0'))
       # print(x2.get_device())
        
        features_for_validation=torch.tensor(features_for_validation).float()
        (givenVal,predictVal,answerVal) = self.sliding_window(features_for_validation)

        #size=int(math.ceil(x2.shape[0]/batchsize))

        for i in range(steps):
            guess = self.GenModel(self.GenModelEnc, self.GenModelDec,dataset,batchsize)
            guess = torch.cat(guess, dim=0)
            x3 = x2 + (mask * guess)
            (given,predict,answer) = self.sliding_window(x3)
            (gradgiven, gradpredict, answergrad, loss) = ObtainGradientofData(given, predict, answer, givenVal, predictVal, answerVal, self.TargetModelEnc, self.TargetModelDec, epochs,weight=1, batch_size=batchsize)
            error_func = ErrorFunctionWithPredefinedLossAndGradNew.apply
            loss1 = error_func(given, predict, answer, loss.requires_grad_(), gradgiven, gradpredict, answergrad)
            self.GenModelEnc_optimizer.zero_grad()
            self.GenModelDec_optimizer.zero_grad()
            loss1.backward()
            self.GenModelEnc_optimizer.step()
            self.GenModelDec_optimizer.step()
            print("Loss for generator = " + str(loss1))
        return x3

    def TrainGenModelWithHackedFeatures(self, x2, features_for_validation, epochs=50, batchsize=400, weight=0.5, InverseLossFunction=True, steps=50):
        
        x2 = torch.tensor(x2).float()
        features_for_validation=torch.tensor(features_for_validation).float()
        
        #a = x2.shape[0]-100
        #x2 = x2[0:a]
        #x2 = x2.to(torch.device('cuda:0'))
        mask1 = self.MaskForAttackPoints(x2.shape)
        mask = torch.tensor(mask1)
        #mask2 = self.MaskForAttackPoints(features_for_validation.shape)
        #maskVal = torch.tensor(mask2)
        x2 = x2 * mask
        x2 = x2.to(torch.device('cuda:0'))
        #print(x2[:5])
        #features_for_validation = features_for_validation * maskVal
        #features_for_validation = features_for_validation.to(torch.device('cuda:0'))
        data_split = self.split_data(x2)
        dataset = PoisonedDataset(data_split)


       # print(x2.get_device())
        
        
        (givenVal,predictVal,answerVal) = self.sliding_window(features_for_validation)

        #size=int(math.ceil(x2.shape[0]/batchsize))
        a = x2.shape[0]-100
        x2 = x2[0:a]
        #x2 = x2.to(torch.device('cuda:0'))
        mask3 = self.MaskForAttackPoints(x2.shape)
        mask4 = torch.tensor(mask3).to(torch.device('cuda:0'))

        for i in range(steps):
            guess = self.GenModel(self.GenModelEnc, self.GenModelDec,dataset,batchsize)
            guess = torch.cat(guess, dim=0)
            #print(guess[:5])
            x3 = x2 + (mask4 * guess)
            #print(x3[:5])
            (given,predict,answer) = self.sliding_window(x3)
            (gradgiven, gradpredict, answergrad, loss) = ObtainGradientofData(given, predict, answer, givenVal, predictVal, answerVal, self.TargetModelEnc, self.TargetModelDec, epochs, batch_size=batchsize)
            error_func = ErrorFunctionWithPredefinedLossAndGradNew.apply
            loss1 = error_func(given, predict, answer, loss.requires_grad_(), gradgiven, gradpredict, answergrad)
            self.GenModelEnc_optimizer.zero_grad()
            self.GenModelDec_optimizer.zero_grad()
            loss=weight*loss1+(1-weight)*GetLoss(given, predict, answer, self.TargetModelEnc, self.TargetModelDec)
            loss.backward()
            self.GenModelEnc_optimizer.step()
            self.GenModelDec_optimizer.step()
            print("Loss for generator = " + str(loss1))
        return x3        
