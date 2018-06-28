# Implement a set of BASIC Vision Modules

import torch.nn as nn
import torch

def image_to_vector_module(image_size,*args):
	'''
	Return a classical sequence of convolution+batchnorm aiming at transforming an image to a vector
	@return module,size of the output vectors
	'''

	class Convolution_MNIST(nn.Module):
		'''
		input size is 1x28x28 (MNIST)
		output size is 10x3x3
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
			self.os=0

		def forward(self,x):
			x = self.convol(x)	
			if (self.os==0):
				self.os=1
				for i in range(1,len(x.size())):
					self.os*=x.size()[i]
			x=x.view(-1,self.os)		
			return x

	class Convolution_MNIST_56(nn.Module):
		'''
		input size is 1x28x28 (MNIST)
		output size is 10x3x3
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
			self.os=0

		def forward(self,x):
			x = self.convol(x)
			if (self.os==0):
				self.os=1
				for i in range(1,len(x.size())):
					self.os*=x.size()[i]
			x=x.view(-1,self.os)		
			return x



	image_z=image_size[0]
	sx=image_size[1]
	sy=image_size[2]
	print(image_size)
	if ((image_z==1) and (sx==28) and (sy==28)):
		return Convolution_MNIST(),90
	if ((image_z==1) and (sx==56) and (sy==56)):
		return Convolution_MNIST_56(),90


	

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
       
	
 
