import numpy as np
import torch
from torch import nn, optim
from PIL import Image
import sys
import pickle
import poisoning_attack_module

def attack_feature(sensorID,num):
	l=list()
	for i in range(0,51*num,51):
		if sensorID==0:
			for j in range(0,5):
				l.append(i+j)
		if sensorID==1:
			for j in range(5,16):
				l.append(i+j)
		if sensorID==2:
			for j in range(16,25):
				l.append(i+j)
		if sensorID==3:
			for j in range(25,34):
				l.append(i+j)
		if sensorID==4:
			for j in range(34,47):
				l.append(i+j)
		if sensorID==5:
			for j in range(47,51):
				l.append(i+j)
	return l


def main():
	step=5000
	param = sys.argv

	training_data=np.loadtxt(param[1], delimiter=',').astype(np.float32)
	
	with open(param[2], 'rb') as i:
		cl = pickle.load(i)
		
	li_attack=attack_feature(int(param[3]),4)
	perm = np.random.permutation(training_data.shape[0])
	

	for i in range(0,10):
#	for i in range(0,training_data.shape[0],step):
		Attacker=poisoning_attack_module.DirectPoisoningAttacker(cl,li_attack)
		if "robust" in param:
			Attacker=poisoning_attack_module.DirectPoisoningAttacker(cl.model,li_attack)
#		data=Attacker.Generate(training_data[perm[i:i+step]],training_data[0:step],None,None,InverseLossFunction=True)
		data=Attacker.Generate(training_data,training_data,None,None,InverseLossFunction=True)
		for j in range(10):
			cl.update(data)
		with open(param[2]+"-Target-"+param[3]+"-"+str(i)+".pkl", 'wb') as o:
			pickle.dump(cl, o)


	
if __name__ == '__main__':
    main()