import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.autograd import Variable
from ConvLSTM import *

#residual block without batch normalization
class ResBlock(nn.Moudle):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=5, stride=1, padding=2)
        self.relu = nn.ReLU(inplace=True)#attrntion:relu not prelu
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=5, stride=1, padding=2)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.relu(residual)
        residual = self.conv2(residual)#the original model multipy a value 0.1 here,which is to be analyzed in the feature
        out      = torch.add(x,residual)

        return out


class InBlock(nn.Moudle):
	def __init__(self):
		super(InBlock,self).__init__()
		self.ResBlock1=ResBlock(32)
		self.ResBlock2=ResBlock(32)
		self.ResBlock3=ResBlock(32)

	def forward(self,x):
		x=f.relu(self.ResBlock1(x))
		x=f.relu(self.ResBlock2(x))
		x=f.relu(self.ResBlock3(x))

		return x


class OutBlock(nn.Moudel):
	def __init__(self):
		super(OutBlock,self).__init__()
		self.ResBlock1=ResBlock(32)
		self.ResBlock2=ResBlock(32)
		self.ResBlock3=ResBlock(32)
		self.conv=nn.Conv2d(32,3,5,1,2)

	def forward(self,x):
		x=f.relu(self.ResBlock1(x))
		x=f.relu(self.ResBlock2(x))
		x=f.relu(self.ResBlock3(x))
		x=f.relu(self.conv(x))

		return x	


class EBlock(nn.Moudel):
	def __init__(self,in_channels):
		super(EBlock,self).__init__()
		self.conv=nn.Conv2d(in_channels,in_channels*2,5,2,2)#downsampling and extend channels
		self.ResBlock1=ResBlock(in_channels*2)
		self.ResBlock2=ResBlock(in_channels*2)
		self.ResBlock3=ResBlock(in_channels*2)

	def forward(self,x):
		x=f.relu(self.conv(x))
		x=f.relu(self.ResBlock1(x))
		x=f.relu(self.ResBlock2(x))
		x=f.relu(self.ResBlock3(x))

		return x


class DBlock(nn.Moudle):
	def __init__(self,in_channels):
		super(DBlock,self).__init__()
		self.ResBlock1=ResBlock(in_channels)
		self.ResBlock2=ResBlock(in_channels)
		self.ResBlock3=ResBlock(in_channels)
		self.conv = nn.ConvTranspose2d(in_channels, in_channels/2, 5, 2, 2)#upsample and reduce channels  #or 4,2,2

	def forward(self,x):
		x=f.relu(self.ResBlock1(x))
		x=f.relu(self.ResBlock2(x))
		x=f.relu(self.ResBlock3(x))
		x=f.relu(self.conv(x))

		return x



class Scale_Recurrent_Network(nn.Moudle):
	def __init__(self):
		super(Single_Scale_Recurrent_Network,self).__init__()
		self.in1=nn.Conv2d(3,32,5,1,2)
		self.in2=nn.Conv2d(6,32,5,1,2)

		self.convlstm=ConvLSTM(input_size=(64,64),
			input_dim=128,
			hidden_dim=128,
			kernel_size=(5,5),
			num_layers=1,
			batch_first=True,
			bias=True,
			return_all_layers=False
			)
		self.inblock=InBlock()
		self.eblock1=EBlock(32)
		self.eblock2=EBlock(64)
		self.dblock1=DBlock(128)
		self.dblock2=DBlock(64)
		self.outblock=OutBlock()

	def forward(self,x,state=None):
		if state is not None:
			x=self.in2(x)
		else:
			x=self.in1(x)

		x1=self.inblock(x)
		x2=self.eblock1(x1)
		x3=self.eblock2(x2)
		hidden,state=self.convlstm(x3,state)
		x4=self.dblock1(hidden)
		x4=torch.add(x2,x4)#skip connection between eblock and deblock
		x5=self.dblock2(x4)
		x5=torch.add(x1,x5)#skip connection between inblock and outblock
		output=self.outblock(x5)

		return output,state
