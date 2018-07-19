# This module implements different functions for loading/reading datasets
# All the functions return an array of mini-batches


import logging
import torchvision
import random
from torchvision import datasets, transforms
import torch.utils.data

def extract_coco_images(cocodir,cocoinstances,categories=[],proportion=1.0):
	'''
	Exract images from coco given a list of categories. Return is list of [(image, annotations)]
	:param cocodir:
	:param cocoinstances:
	:param categories:
	:param proportion:
	:return:
	'''
	transform=transforms.Compose([transforms.Resize((28*4,28*4)),transforms.ToTensor()])
	train_dataset=datasets.CocoDetection(cocodir,cocoinstances,transform=transform)


	#Read categories
	CATEGORIES_IDS_TO_KEEP=[]
	for c in train_dataset.coco.dataset["categories"]:
		if (c['name'] in categories):
			CATEGORIES_IDS_TO_KEEP.append(c['id'])

	nb_kept=0
	nb_total=0
	images=[]
	for idx in range(len(train_dataset)):
		y=train_dataset.ids[idx]
		y = train_dataset.coco.getAnnIds(imgIds=y)
		y = train_dataset.coco.loadAnns(y)
		ids=[]
		for a in y:
			if (not a["category_id"] in ids):
				ids.append(a["category_id"])
		keep=False
		for c in ids:
			if (c in CATEGORIES_IDS_TO_KEEP):
				keep=True

		if (keep):
			if (random.random()<proportion):
				x, z = train_dataset[idx]
				img_id = train_dataset.ids[idx]
				path = train_dataset.coco.loadImgs(img_id)[0]['file_name']
				images.append((x,ids))
				print(ids)



	logging.info("found %d images"%len(images))
	return(images)

def load_unsupervised_images(images,size_batches=64,nb_batches=10):
	'''
	Takes a list of images and build batches
	:param images:
	:param size_batches:
	:param nb_batches:
	:return:
	'''
	batches=[]
	for n in range(nb_batches):
		batch=torch.Tensor(images[0].size())
		batch=batch.unsqueeze(0)
		batch=batch.repeat(size_batches,1,1,1)
		for i in range(size_batches):
			idx=random.randint(0,len(images)-1)
			batch[i]=images[idx]
		batches.append(batch)
	return batches


def load_supervised_images(images,size_batches=64,nb_batches=10):
	'''
	Takes a list of images and categories and build batches. Categories are indexes between 0 and C-1
	:param images:
	:param size_batches:
	:param nb_batches:
	:return:
	'''
	batches=[]
	for n in range(nb_batches):
		batch=torch.Tensor(images[0][0].size())
		batch_y = torch.Tensor(size_batches)

		batch=batch.unsqueeze(0)
		batch=batch.repeat(size_batches,1,1,1)
		for i in range(size_batches):
			idx=random.randint(0,len(images)-1)
			batch[i]=images[idx][0]
			print(images[idx][1])
			batch_y[i] = images[idx][1]

		batches.append((batch,batch_y))
	return batches


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

def transform_image_to_puzzle(image, nbx,nby):
	'''
	Transform an B*W*H image into a table of randomly shuffle pieces
	'''
	pieces=[]
	sx=image.size()[1]/nbx
	assert int(sx)==sx
	sy=image.size()[2]/nby
	assert sy==int(sy)
	
	for x in range(nbx):
		for y in range(nby):
			p=image.narrow(1,int(x*sx),int(sx)).narrow(2,int(y*sy),int(sy))
			pieces.append(p)
	
	random.shuffle(pieces)
	return(pieces)

def load_puzzle_mnist(size_batches=64,sampling_rate=1.0,nb_pieces_x=1,nb_pieces_y=1):
	'''	
	'''
	
	logging.info("Creating mini-batches from MNIST PUZZLE")
	loader = torch.utils.data.DataLoader(
    datasets.MNIST(root='.', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])), batch_size=size_batches, shuffle=True, num_workers=0)
	
	all_examples=[]	
	for batch_idx, (data, target) in enumerate(loader):
		if (random.random()<sampling_rate):
			batch_pieces=None		
			for i in range(data.size()[0]):			
				pieces=transform_image_to_puzzle(data[0],nb_pieces_x,nb_pieces_y)
				if (batch_pieces is None):					
					batch_pieces=pieces[0].unsqueeze(0).unsqueeze(0).repeat(len(pieces),size_batches,1,1,1)
				for k in range(len(pieces)):
					batch_pieces[k][i]=pieces[k]
			
			all_examples.append((batch_pieces,data.clone()))

	logging.info("\t%d batches of size %d built"%(len(all_examples),size_batches))
	return all_examples



def load_supervised_mnist(size_batches=64, sampling_rate=1.0):
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

	all_examples = []
	for batch_idx, (data, target) in enumerate(loader):
		if (random.random() < sampling_rate):
			all_examples.append((data,target))

	logging.info("\t%d batches of size %d built" % (len(all_examples), size_batches))
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
	
def load_supervised_mnist_random_position(size_batches=64,sampling_rate=1.0):
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
				
			
			all_examples.append((batch,target.clone()))

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
		

def sample_batches(batches,proportion):
    retour=[]
    for aa in batches:
        if (random.random()<proportion):
            retour.append(aa)

    return retour

   


   

