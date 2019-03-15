# Implement a set of BASIC Vision Modules

import torch.nn as nn
import torch
from torch.autograd import Function

def inverse_batch_of_2x2matrices(A):
	device=A.device
	swap=torch.LongTensor([1,0]).to(device)
	mask=torch.Tensor([[1,-1],[-1,1]]).to(device)
	mask=mask.repeat(A.size()[0],1,1)
			
	A2=torch.index_select(A,1,swap)
	A2=torch.index_select(A2,2,swap)
	A2=A2.transpose(1,2)
	A2=A2*mask
	
	mul=torch.bmm(A,A2)
	mul2=torch.index_select(mul,1,swap)
	mul=mul+mul2
	mul=1.0/mul
	Ainverse=mul*A2
	return Ainverse


def inverse_stn(stn_matrix):
	'''
	Get a Bx2x3 set of grids, and returns grids corresponding to the inverse transformation
	:param stn_matrix:
	:return:
	'''
	A=stn_matrix.narrow(2,0,2)
	B=stn_matrix.narrow(2,2,1)
	#inverse_A = [t.inverse() for t in torch.functional.unbind(A)]
	#inverse_A = torch.stack(inverse_A, dim=0)
	inverse_A=inverse_batch_of_2x2matrices(A)
	inverse_AB = torch.bmm(inverse_A, B)
	inverse_stn = torch.cat([inverse_A, -inverse_AB], dim=2)
	return inverse_stn


class GradReverse(Function):
    def __init__(self, lambd):
        self.lambd = lambd

    def forward(self, x):
        return x.view_as(x)

    def backward(self, grad_output):
        return (grad_output * -self.lambd)

def grad_reverse(x, lambd=1.0):
    return GradReverse(lambd)(x)



class STN_Square(nn.Module):
    #Takes a Bx1 localisation + Bx2 translation and returns a Bx2x3 grid
    def __init__(self):
        super(STN_Square, self).__init__()

        self.mask=torch.ones(2,2)
        self.mask[0][1]=0
        self.mask[1][0]=0
        self.mask=self.mask.unsqueeze(0)

    # Spatial transformer network forward function
    def forward(self, localization,translation):
        dm=self.mask.device
        ds=localization.device
        if (dm!=ds):
            self.mask=self.mask.to(ds)

        localization=localization.unsqueeze(-1).repeat(1,1,2)

        if (self.mask.size()[0]!=localization.size()[0]):
            self.mask=self.mask.repeat(localization.size()[0],1,1)

        localization=localization*self.mask

#
        theta = torch.cat([localization,translation.unsqueeze(-1)],dim=2)
        return theta

class Coords_To_Affine(nn.Module):
    #Coords are between -1 and +1
	#Assuming that image_size is an int (square images)
	def __init__(self):
		super(Coords_To_Affine, self).__init__()
		self.mask=torch.ones(2,2)
		self.mask[0][1]=0
		self.mask[1][0]=0
		self.mask=self.mask.unsqueeze(0)

		# Spatial transformer network forward function
	#Coords is a Bx2 vector => (x1,y1) (x2,y2)
	def forward(self, xy1,xy2):

		dm = self.mask.device
		ds = xy1.device
		if (dm != ds):
			self.mask = self.mask.to(ds)

		B=(xy1+xy2)/2

		A=(xy2-xy1)/2
		A=A.unsqueeze(-1).repeat(1,1,2)
		A=A*self.mask
		AB=torch.cat([A, B.unsqueeze(-1)], dim=2)

		return AB

class MNIST28_To_Vector(nn.Module):
	'''
	input size is 1x28x28 (MNIST)
	output size is 90
	'''

	def __init__(self):
		nn.Module.__init__(self)
		self.convol = nn.Sequential(
		nn.BatchNorm2d(1),
		nn.Conv2d(1, 8, kernel_size=7),
		nn.MaxPool2d(2, stride=2),
		nn.ReLU(True),
		nn.BatchNorm2d(8),
		nn.Conv2d(8, 10, kernel_size=5),
		nn.MaxPool2d(2, stride=2),
		nn.ReLU(True),
		)
		self.os = 0

	def forward(self, x):
		x = self.convol(x)
		if (self.os == 0):
			self.os = 1
			for i in range(1, len(x.size())):
				self.os *= x.size()[i]
		x = x.view(-1, self.os)
		return x


class MNIST14_To_Vector(nn.Module):
	'''
	input size is 1x14x14 (MNIST)
	output size is 10
	'''

	def __init__(self):
		nn.Module.__init__(self)
		self.convol = nn.Sequential(
		nn.BatchNorm2d(1),
		nn.Conv2d(1, 8, kernel_size=5),
		nn.MaxPool2d(2, stride=2),
		nn.ReLU(True),
		nn.BatchNorm2d(8),
		nn.Conv2d(8, 10, kernel_size=3),
		nn.MaxPool2d(2, stride=2),
		nn.ReLU(True),
		)
		self.os = 0

	def forward(self, x):
		x = self.convol(x)
		if (self.os == 0):
			self.os = 1
			for i in range(1, len(x.size())):
				self.os *= x.size()[i]
		x = x.view(-1, self.os)
		return x


class MNIST56_To_Vector(nn.Module):
	'''
	input size is 1x56x56 (MNIST)
	output size is 90
	'''

	def __init__(self):
		nn.Module.__init__(self)
		self.convol = nn.Sequential(
		nn.BatchNorm2d(1),
		nn.Conv2d(1, 8, kernel_size=7),
		nn.MaxPool2d(2, stride=2),
		nn.ReLU(True),
		nn.BatchNorm2d(8),
		nn.Conv2d(8, 10, kernel_size=5),
		nn.MaxPool2d(2, stride=2),
		nn.ReLU(True),
		nn.BatchNorm2d(10),
		nn.Conv2d(10, 10, kernel_size=5),
		nn.MaxPool2d(2, stride=2),
		nn.ReLU(True)
		)
		self.os = 0

	def forward(self, x):
		x = self.convol(x)
		if (self.os == 0):
			self.os = 1
			for i in range(1, len(x.size())):
				self.os *= x.size()[i]
		x = x.view(-1, self.os)
		return x
	
class COCO112_To_Vector(nn.Module):
	'''
	input size is 1x28x28 (MNIST)
	output size is 1000
	'''

	def __init__(self):
		nn.Module.__init__(self)
		self.convol = nn.Sequential(
			nn.BatchNorm2d(3),
			nn.Conv2d(3, 8, kernel_size=7),
			nn.MaxPool2d(2, stride=2),
			nn.ReLU(True),
			nn.BatchNorm2d(8),
			nn.Conv2d(8, 10, kernel_size=5),
			nn.MaxPool2d(2, stride=2),
			nn.ReLU(True),
			nn.BatchNorm2d(10),
			nn.Conv2d(10, 10, kernel_size=5),
			nn.MaxPool2d(2, stride=2),
			nn.ReLU(True)
		)
		self.os = 0

	def forward(self, x):
		x = self.convol(x)
		if (self.os == 0):
			self.os = 1
			for i in range(1, len(x.size())):
				self.os *= x.size()[i]
		x = x.view(-1, self.os)
		return x
	
 
class COCO14_To_Vector(nn.Module):
	def __init__(self):
		nn.Module.__init__(self)
		self.convol = nn.Sequential(
		nn.BatchNorm2d(3),
		nn.Conv2d(3, 8, kernel_size=5),
		nn.MaxPool2d(2, stride=2),
		nn.ReLU(True),
		nn.BatchNorm2d(8),
		nn.Conv2d(8, 10, kernel_size=3),
		nn.MaxPool2d(2, stride=2),
		nn.ReLU(True),
		)
		self.os = 0

	def forward(self, x):
		x = self.convol(x)
		if (self.os == 0):
			self.os = 1
			for i in range(1, len(x.size())):
				self.os *= x.size()[i]
		x = x.view(-1, self.os)
		return x
	
 
class COCO28_To_Vector(nn.Module):
	def __init__(self):
		nn.Module.__init__(self)
		self.convol = nn.Sequential(
		nn.BatchNorm2d(3),
		nn.Conv2d(3, 8, kernel_size=5),
		nn.MaxPool2d(2, stride=2),
		nn.ReLU(True),
		nn.BatchNorm2d(8),
		nn.Conv2d(8, 10, kernel_size=3),
		nn.MaxPool2d(2, stride=2),
		nn.ReLU(True),
		)
		self.os = 0

	def forward(self, x):
		x = self.convol(x)
		if (self.os == 0):
			self.os = 1
			for i in range(1, len(x.size())):
				self.os *= x.size()[i]
		x = x.view(-1, self.os)
		return x
	
 
