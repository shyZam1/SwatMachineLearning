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

def GetLossFunc(batch, enc, dec, autograd=True):
    if autograd:
        given = batch['given'].to(torch.device('cuda:0'))
        predict = batch['predict'].to(torch.device('cuda:0'))
        answer = batch['answer'].to(torch.device('cuda:0')) 
        given1 = given.clone().detach().requires_grad_(True)
        predict1 = predict.clone().detach().requires_grad_(True)
        encoder_outs, context = enc(given1)
        guess = dec(encoder_outs, context, predict1)
        mse_loss = nn.MSELoss()
        loss = mse_loss(guess, answer)
    else:
        given1 = batch['given'].to(torch.device('cuda:0'))
        predict1 = batch['predict'].to(torch.device('cuda:0'))
        answer = batch['answer'].to(torch.device('cuda:0')) 
        encoder_outs, context = enc(given1)
        guess = dec(encoder_outs, context, predict1)
        mse_loss = nn.MSELoss()
        loss = mse_loss(guess, answer)
        
    return (loss,given1, predict1 )

def ObtainGradientOfWeightInModel(Model,Data,Label,ListOfUnusedFeatures=None,InverseLossFunction=False,Max=1.0,epsilon=1e-4,MaxGrad=100.0):
    m=copy.deepcopy(Model)
    m.zero_grad()
    #loss_func = m.get_loss_func()
    if not InverseLossFunction:
        #loss = loss_func(F.clipped_relu(Data,z=Max), Label,normalize=False)
        loss = m.get_loss_func(torch.clamp(Data, z=Max), Label)
    else:
        loss_tmp = m.get_loss_func(torch.clamp(Data, z=Max), Label)
        loss = 1/(loss_tmp+epsilon)

    loss.backward()
    for w in m.parameters():
        if np.max(np.abs(w.grad))>MaxGrad:
            w.grad=w.grad/np.max(np.abs(w.grad))*MaxGrad
    return (loss.data,m)
#--------------------------------------------------------------------------------------------------------------

def ObtainGradientOfWeightInModelNew(ModelEnc, ModelDec, TrainingData, batch_size, MaxGrad=100.0):
    enc = copy.deepcopy(ModelEnc)
    dec = copy.deepcopy(ModelDec)
    enc.zero_grad()
    dec.zero_grad()
    epoch_loss = 0

    for batch in DataLoader(TrainingData, batch_size=batch_size, shuffle=True):
        #answer, guess = Network.infer(batch)
        #loss = Network.get_loss_func(guess, answer)
        #given = batch['given'].to(torch.device('cuda:0'))
        #predict = batch['predict'].to(torch.device('cuda:0'))
        #answer = batch['answer'].to(torch.device('cuda:0')) 
        #encoder_outs, context = enc(given)
        #guess = dec(encoder_outs, context, predict)
        #mse_loss = nn.MSELoss()
        #loss = mse_loss(guess, answer)
        (loss, given, predict) = GetLossFunc(batch, enc, dec, autograd=False)
        loss.backward(retain_graph=True)
        #epoch_loss += loss.item()
        epoch_loss += loss
    
    for w in enc.parameters():
        if torch.max(torch.abs(w.grad))>MaxGrad:
            w.grad = w.grad/torch.max(torch.abs(w.grad))*MaxGrad
    
    for w in dec.parameters():
        if torch.max(torch.abs(w.grad))>MaxGrad:
            w.grad = w.grad/torch.max(torch.abs(w.grad))*MaxGrad
    
    return (epoch_loss, enc, dec)
#--------------------------------------------------------------------------------------------------------------
    
def ObtainListOfParameters(Model):
    l=list()
    for w in Model.parameters():
        l.append(w)
        #l.append(copy.deepcopy(w))#using deepcopy I get an error whereby the grad output is none
    return l

def UpdateModel(Model, Alpha,MaxGrad=1,ListOfParam=None):
	if ListOfParam is None:
		for w in Model.parameters():
			w.data=w.data + Alpha* w.grad
	else:
		i=0
		for w in Model.parameters():
			w.data=w.data +Alpha * ListOfParam[i].grad
			i=i+1

def TrainTargetModel(Model, TrainingData, TrainingLabel,NumIteration=200,Alpha=0.1,DataMax=1.0):
	m=copy.deepcopy(Model)
	for t in range(NumIteration):
		(loss,m)=ObtainGradientOfWeightInModel(m,TrainingData,TrainingLabel,Max=DataMax)
		UpdateModel(m,-Alpha)
	return (loss,m)

#-------------------------------------------------------------------------------------------------------

def TrainTargetModelNew(ModelEnc, ModelDec, TrainingData, epochs, batch_size, Alpha=0.1):
    enc = copy.deepcopy(ModelEnc)
    dec = copy.deepcopy(ModelDec)

    for e in range(epochs):
        (epoch_loss,enc,dec) = ObtainGradientOfWeightInModelNew(enc,dec, TrainingData, batch_size)
        UpdateModel(enc,-Alpha)
        UpdateModel(dec,-Alpha)
        print("Train Target Model New epoch loss ="+ str(epoch_loss))
    return (epoch_loss,enc,dec)

#---------------------------------------------------------------------------------------------------------

# this is the function that is trying to maximize the loss function to generate the attack , we are not using any optimization function to calculate the loss 
def ObtainGradientOfData(TrainedModel,Xp,Xval,Yp,Yval,NumIteration=3,Alpha=0.01,Epsilon=1e-6,max_grad=100,Mask=None,ListOfUnusedFeatures=None,InverseLossFunction=True):
    dxp=np.zeros(Xp.shape).astype(np.float32)
    (loss,m)=TrainTargetModel(TrainedModel, Xp,Yp,NumIteration=NumIteration,Alpha=Alpha)
    (loss,m)=ObtainGradientOfWeightInModel(m,Xval,Yval,ListOfUnusedFeatures=ListOfUnusedFeatures,InverseLossFunction=InverseLossFunction)
    w=ObtainListOfParameters(m)
    for i in range(NumIteration-1):
        (loss_tmp,m)=ObtainGradientOfWeightInModel(m,Xp,Yp)
        UpdateModel(m,Alpha)
        M1=copy.deepcopy(m)
        M1.zero_grad()
        UpdateModel(M1,(0.5)*Epsilon,ListOfParam=w)
        X1=copy.deepcopy(Xp) #stuck here 
        torch.autograd.grad([M1.get_loss_func(torch.clamp(X1,z=1.0),Yp)], [X1])
        M1.zero_grad()
        loss_tmp = M1.get_loss_func(torch.clamp(X1,z=1.0),Yp)
        loss_tmp.backward()
        m1=ObtainListOfParameters(M1)
        M2=copy.deepcopy(m)
        M2.zero_grad()
        UpdateModel(M2,(-0.5)*Epsilon,ListOfParam=w)
        X2=copy.deepcopy(Xp)
        torch.autograd.grad([M2.get_loss_func(torch.clamp(X2,z=1.0),Yp)], [X2])
        M2.zero_grad()
        loss_tmp = M2.get_loss_func(torch.clamp(X2,z=1.0),Yp)
        loss_tmp.backward()
        m2=ObtainListOfParameters(M2)
        
        ddxp=(X1.grad-X2.grad)/Epsilon
        if Mask is not None:
            ddxp=ddxp*Mask
        dxp = dxp - Alpha*ddxp
        for j in range(len(m1)):
            w_tmp=(m1[j].grad-m2[j].grad)/Epsilon
            w[j].grad=w[j].grad-Alpha*w_tmp
    if np.max(np.abs(dxp))>max_grad:
        dxp=dxp/np.max(np.abs(dxp))*max_grad
    return dxp,loss

#------------------------------------------------------------------------------------------------------------------
def CreateZeroTensor(TrainingData, batch_size):
    givenzeros = []
    predictzeros = []
    for batch in DataLoader(TrainingData, batch_size=batch_size, shuffle=True):
        given = batch['given'].to(torch.device('cuda:0'))
        predict = batch['predict'].to(torch.device('cuda:0'))
        dgiven = torch.zeros(given.shape).to(torch.device('cuda:0'))
        dpredict = torch.zeros(predict.shape).to(torch.device('cuda:0'))
        givenzeros.append(dgiven)
        predictzeros.append(dpredict)
    
    return (givenzeros, predictzeros)


def CalculateLoss(enc, dec, TrainingData, batch_size):
    for batch in DataLoader(TrainingData, batch_size=batch_size, shuffle=True):
        (loss_temp, given, predict) = GetLossFunc(batch, enc, dec, autograd=False)
        loss_temp.backward()
    
    return loss_temp

def GetGradientUsingAutoGrad(enc, dec, TrainingData,batch_size):
    givenbag = []
    predictbag = []
    for batch in DataLoader(TrainingData, batch_size=batch_size, shuffle=True):
        (loss, given1, predict1) = GetLossFunc(batch, enc, dec, autograd=True)
        grad = torch.autograd.grad(outputs=loss, inputs=(given1,predict1), retain_graph=True, allow_unused=True)
        loss.backward()
        givenbag.append(given1.grad)
        predictbag.append(predict1.grad)
    return (grad,enc, dec, givenbag, predictbag)
#------------------------------------------------------------------------------------------------------------------
def ObtainGradientofDataNew(Encoder, Decoder, TrainingData, epochs,batch_size ,Alpha=0.01, Epsilon=1e-6, max_grad=100, Mask=None, ListofUnusedFeatures=None, InverseLossFunction=True):
    givenbasket = []
    predictbasket = []
    gradgiven = []
    gradpredict = []
    (givenzeros, predictzeros) = CreateZeroTensor(TrainingData=TrainingData, batch_size=batch_size)
    giventemp = givenzeros
    predicttemp = predictzeros
    givengrad = givenzeros
    predictgrad = predictzeros
    
    (epoch_loss,enc,dec)=TrainTargetModelNew(Encoder, Decoder, TrainingData=TrainingData, epochs=epochs, batch_size=batch_size, Alpha=Alpha)
    (epoch_loss,enc,dec)=ObtainGradientOfWeightInModelNew(enc,dec,TrainingData, batch_size=batch_size)
    w1=ObtainListOfParameters(enc)
    w2=ObtainListOfParameters(dec)
    for i in range(epochs-1):
        (epoch_loss_temp,enc,dec,)=ObtainGradientOfWeightInModelNew(enc,dec,TrainingData, batch_size=batch_size)
        UpdateModel(enc,Alpha)
        UpdateModel(dec,Alpha)
        ENC1=copy.deepcopy(enc)
        DEC1=copy.deepcopy(dec)
        ENC1.zero_grad()
        DEC1.zero_grad()
        UpdateModel(ENC1,(0.5)*Epsilon,ListOfParam=w1)
        UpdateModel(DEC1,(0.5)*Epsilon,ListOfParam=w2)
        (grad,ENC1,DEC1, given1, predict1)=GetGradientUsingAutoGrad(ENC1, DEC1, TrainingData=TrainingData, batch_size=batch_size)
        #print(grad)
        ENC1.zero_grad()
        DEC1.zero_grad()
        epoch_loss_temp = CalculateLoss(ENC1, DEC1, TrainingData, batch_size=batch_size)
        #epoch_loss_temp.backward()
        enc1=ObtainListOfParameters(ENC1)
        dec1=ObtainListOfParameters(DEC1)
        ENC2=copy.deepcopy(enc)
        DEC2=copy.deepcopy(dec)
        ENC2.zero_grad()
        DEC2.zero_grad()
        UpdateModel(ENC2,(-0.5)*Epsilon,ListOfParam=w1)
        UpdateModel(DEC2,(-0.5)*Epsilon,ListOfParam=w2)
        (grad,ENC2,DEC2, given2, predict2)=GetGradientUsingAutoGrad(ENC2, DEC2, TrainingData=TrainingData, batch_size=batch_size)
        ENC2.zero_grad()
        DEC2.zero_grad()
        epoch_loss_temp = CalculateLoss(ENC2, DEC2, TrainingData, batch_size=batch_size)
        #epoch_loss_temp.backward()
        enc2=ObtainListOfParameters(ENC2)
        dec2=ObtainListOfParameters(DEC2)
        for i in range(len(given1)):
            giventemp[i] = (given1[i]-given2[i])/Epsilon
            predicttemp[i] = (predict1[i]-predict2[i])/Epsilon
           # givenbasket.append((given1[i]-given2[i])/Epsilon) #same as ddxp in sensei's code
           # predictbasket.append((predict1[i]-predict2[i])/Epsilon) #same as ddxp in sensei's code
        #print('-------------len of givenzeros---------------')
        #print(len(givenzeros))
        #print('--------------len of givenbasket---------------')
        #print(len(giventemp))
        for i in range(len(giventemp)):
            givengrad[i] = givenzeros[i]-(Alpha*giventemp[i])
            predictgrad[i] = predictzeros[i]-(Alpha*predicttemp[i])
            #gradgiven.append((givenzeros[i]-(Alpha*givenbasket[i])))
            #gradpredict.append((predictzeros[i]-(Alpha*predictbasket[i])))
        
        for j in range(len(enc1)):
            w1_tmp=(enc1[j].grad - enc2[j].grad)/Epsilon
            w1[j].grad = w1[j].grad - Alpha*w1_tmp
        
        for j in range(len(dec1)):
            w2_tmp=(dec1[j].grad - dec2[j].grad)/Epsilon
            w2[j].grad = w2[j].grad - Alpha*w2_tmp
        #print('---------------reach here---------------')
    for i in range(len(givengrad)):
        if torch.max(torch.abs(givengrad[i]))>max_grad:
            givengrad[i] = givengrad[i]/torch.max(torch.abs(givengrad[i]))*max_grad
            
        if torch.max(torch.abs(predictgrad[i]))>max_grad:
            predictgrad[i] = predictgrad[i]/torch.max(torch.abs(predictgrad[i]))*max_grad
    #print('----------reach last line --------------')
    print(len(givengrad))
    print(givengrad[0])
    print(predictgrad[0])
    return (givengrad, predictgrad, epoch_loss)
        
    
#-----------------------------------------------------------------------------------------------------------------

class ErrorFunctionWithPredefinedLossAndGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx,inputs):
        ctx.save_for_backward((0,1,2))
        loss=inputs[1]
        return loss
    
    @staticmethod
    def backward(ctx, indexes, gy):
        x,y,z = ctx.saved_tensors
        gy0 = torch.broadcast_to(gy[0],y.shape)
        return gy0*z, None, None


def error_with_predefined_loss_and_grad(data,loss,grad):
	return ErrorFunctionWithPredefinedLossAndGrad().apply((data,loss,grad))
    #return ErrorFunctionWithPredefinedLossAndGrad().apply((data,loss,grad))[0]

#------------------------------------------------------------------------------------------------------------------
class ErrorFunctionWithPredefinedLossAndGradNew(torch.autograd.Function):
    @staticmethod
    def forward(ctx,inputs):
        ctx.save_for_backward((0,1,2,3,4))
        #ctx.save_for_backward(given,predict,epochloss,givengrad,predictgrad)
        loss = inputs[2]
        return loss

    @staticmethod
    def backward(ctx,grad_output):
        given,predict,epochloss,givengrad,predictgrad = ctx.saved_tensors

        return givengrad, predictgrad, None, None, None

def error_with_predefined_loss_and_grad_new(given,predict,epochloss,givengrad,predictgrad):
    return ErrorFunctionWithPredefinedLossAndGradNew().apply((given,predict,epochloss,givengrad,predictgrad))

    

#------------------------------------------------------------------------------------------------------------------

def SearchList(li,key):
    for i in range(len(li)):
        if li[i]==key:
            return i
    return -1
#-------------------------------------------------------------------------------------------------------------------
class DirectPoisoningAttackerNew():
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
            #x4 = [x3[i:conf.WINDOW_SIZE + i]]
            x4.append(x3[i:conf.WINDOW_SIZE + i])
    
        split_window = [
                        (w[:conf.WINDOW_GIVEN],
                        w[conf.WINDOW_GIVEN:conf.WINDOW_GIVEN + conf.WINDOW_PREDICT],
                        w[-1]) for w in x4
                    ]
        
        return split_window

    def GenerateNew(self,x2, epochs, batch_size, steps=50):
        #BATCH_SIZE = 4096
        l = nn.Linear(1,x2.size)
        x2 = torch.tensor(x2)
        opt=optim.Adam(l.parameters())
        mask1=self.MaskForAttackPoints(x2.shape)
        mask = torch.tensor(mask1)
        for i in range(steps):
            #x3=self.ClippedRelu(x2+mask*self.ClippedRelu(l(torch.tensor(np.zeros((1,1)).astype(np.float32))),z=1.0).reshape(x2.shape)-1.0*0.5*torch.tensor(np.ones(x2.shape).astype(np.float32))*mask,z=1.0)
            x3=self.ClippedRelu(x2+mask*self.ClippedRelu(l(torch.tensor(np.zeros((1,1)).astype(np.float32))),z=1.0).reshape(x2.shape)-1.0*0.5*torch.tensor(np.ones(x2.shape).astype(np.float32))*mask,z=1.0).float()
            splitdata = self.sliding_window(x3)
            TrainingData = PoisonedDataset(training_data=splitdata)
            (gradgiven, gradpredict,loss) = ObtainGradientofDataNew(self.TargetModelEnc, self.TargetModelDec, TrainingData, epochs, batch_size)
            for batch in DataLoader(TrainingData,batch_size=batch_size, shuffle=True):
                given1 = batch['given'].to(torch.device('cuda:0'))
                predict1 = batch['predict'].to(torch.device('cuda:0'))
                loss1 = error_with_predefined_loss_and_grad_new(given1,predict1,loss,gradgiven,gradpredict)
                #loss1 = poisoning_attack_module.error_with_predefined_loss_and_grad(given1, epoch_loss, gradgiven[i])
                loss1 = Variable(loss1, requires_grad = True)
                #l.cleargrads()
                l.zero_grad()
                loss1.backward()
                #opt.update()
                opt.step()
                print(loss1.data)
        
        return x3

#-------------------------------------------------------------------------------------------------------------------
class DirectPoisoningAttacker():
	def __init__(self, TargetModel,list_of_attack_points):
		self.TargetModel=TargetModel
		self.li=list_of_attack_points

	def ExtractAttackPoints(self, xs):
		l=list()
		(a,b)=xs.shape
		for i in range(len(self.li)):
			l.append(xs[:,self.li[i]].reshape(a,1))
		return torch.hstack(l)
	
	def MaskForAttackPoints(self,size):
		a=np.zeros(size[1])
		for i in self.li:
			a[i]=1
		return np.tile(a,(size[0],1)).astype(np.float32)
			
	def Generate(self,Xp,Xval,Yp,Yval,alpha=0.01,step=50,max_pertubation=1.0,normalizeRequired=True,InverseLossFunction=True):
		if normalizeRequired:
			x2=torch.tensor(self.TargetModel.normalize(Xp))

			xval2=torch.tensor(self.TargetModel.normalize(Xval))
		else:
			x2=Xp
			xval2=Xval
#		x2=F.clipped_relu(x2+(np.random.rand(x2.shape[0],x2.shape[1])-0.5)*2*self.MaskForAttackPoints(x2.shape),z=1.0)
		l=nn.Linear(1,x2.size)
		opt=optim.Adam(l.parameters())
		#opt.setup(l)
		mask=self.MaskForAttackPoints(x2.shape)
		l.b.data=np.random.rand(x2.size).astype(np.float32)*0.5
#		l.b.data=np.zeros(x2.size).astype(np.float32)*0.5
		for i in range(step):
			x3=torch.clamp(x2+mask*torch.clamp(l(torch.tensor(np.zeros((1,1)).astype(np.float32))),z=max_pertubation).reshape(x2.shape)-max_pertubation*0.5*np.ones(x2.shape).astype(np.float32)*mask,z=1.0)
			(d,loss)=ObtainGradientOfData(self.TargetModel,x3,xval2,Yp,Yval,Mask=mask,InverseLossFunction=InverseLossFunction)
			loss1=error_with_predefined_loss_and_grad(x3,loss,d)
			l.cleargrads()
			loss1.backward()
			opt.update()
			print(loss1.data)
		if (normalizeRequired):
			x3=self.TargetModel.denormalize(x3)
		return x3
    #-----------------------------------------------------------------------------------------------------------
    

def ListOfTargetPoint(TargetModel, target):
	out=list(np.argsort(-np.mean(np.abs(TargetModel.test_each(target)),axis=0)))
	return out
