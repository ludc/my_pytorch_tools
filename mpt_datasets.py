# This module implements different functions for loading/reading datasets
# All the functions return an array of mini-batches


import logging
import torchvision
import random
from torchvision import datasets, transforms
import torch.utils.data

def load_unsupervised_mnist(size_batches=64,sampling_rate=1.0):
	'''
	Load batches of numbers from MNIST (train)
	'''
	
	logging.info("Creating mini-batches from MNIST")
	loader = torch.utils.data.DataLoader(
    datasets.MNIST(root='.', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])), batch_size=size_batches, shuffle=True, num_workers=0)
	
	all_examples=[]	
	for batch_idx, (data, target) in enumerate(loader):
		if (random.random()<sampling_rate):
			all_examples.append(data)

	logging.info("\t%d batches of size %d built"%(len(all_examples),size_batches))
	return all_examples
    
def load_unsupervised_mnist_random_position(size_batches=64,sampling_rate=1.0):
	'''
	Load batches of numbers from MNIST that are randomly localted in a  black 56x56 image
	'''
	
	logging.info("Creating mini-batches from MNIST (digit random position)")
	loader = torch.utils.data.DataLoader(
    datasets.MNIST(root='.', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])), batch_size=size_batches, shuffle=True, num_workers=0)
	
	all_examples=[]	
	for batch_idx, (data, target) in enumerate(loader):
		if (random.random()<sampling_rate):
			batch=torch.zeros(data.size()[0],1,56,56)			
			for i in range(data.size()[0]):			
				px=random.randint(0,27)
				py=random.randint(0,27)
				batch[i].narrow(1,px,28).narrow(2,py,28).copy_(data[i])				
			
			all_examples.append(batch)

	logging.info("\t%d batches of size %d built"%(len(all_examples),size_batches))
	return all_examples
    
def load_unsupervised_mnist_two_digits(size_batches=64,nb_batches=100,digits=[0,1]):
	'''
	Create 56x56 images that contains two 14x14 digits 
	'''
	
	logging.info("Creating mini-batches from MNIST (digit random position)")
	loader = torch.utils.data.DataLoader(
    datasets.MNIST(root='.', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.Resize((14,14)),
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,)),
                   ])), batch_size=size_batches, shuffle=True, num_workers=0)
	
	images={}
	
	for batch_idx, (data, target) in enumerate(loader):
		for i in range(target.size()[0]):
			y=target[i].item()
			if (y in digits):
				if (not y in images):
					images[y]=[]				
				images[y].append(data[i])
				
	for k in images:
		logging.info("\tFound %d images for digit %d"%(len(images[k]),k))

	batches=[]
	for n in range(nb_batches):
		x=torch.zeros(size_batches,1,56,56)
		for b in range(size_batches):
			idx1=random.randint(0,len(images[digits[0]])-1)
			idx2=random.randint(0,len(images[digits[1]])-1)
			px1=random.randint(0,56-14-1)
			py1=random.randint(0,56-14-1)
			px2=random.randint(0,56-14-1)
			py2=random.randint(0,56-14-1)
			flag=True
			while(flag):
				flag=False
				overlapx=False
				if ((px2>=px1) and (px2<px1+14)):
					overlapx=True
				if ((px1>=px2) and (px1<px2+14)):
					overlapx=True
				
				overlapy=False
				if ((py2>=py1) and (py2<py1+14)):
					overlapy=True
				if ((py1>=py2) and (py1<py2+14)):
					overlapy=True
				
				if ((overlapx) or (overlapy)):
					flag=True
					px2=random.randint(0,56-14-1)
					py2=random.randint(0,56-14-1)
					#print("%d/%d et %d/%d"%(px1,py1,px2,py2))
			#print("====")
			#print("%d/%d %d/%d"%(px1,py1,px2,py2))
			#print(x.size())
			#print(images[digits[0]][idx1].size())
			#print(images[digits[1]][idx2].size())
			
			x[b].narrow(1,px1,14).narrow(2,py1,14).copy_(images[digits[0]][idx1])	
			x[b].narrow(1,px2,14).narrow(2,py2,14).copy_(images[digits[1]][idx2])	
			
		batches.append(x)
	
	return(batches)
		


   


   

