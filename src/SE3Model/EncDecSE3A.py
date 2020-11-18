import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from e3nn.non_linearities.rescaled_act import sigmoid, swish, tanh
from e3nn.non_linearities.norm_activation import NormActivation
from e3nn.batchnorm import BatchNorm

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from src.SE3Model.convolution import Convolution

def ConvBlock(Rs_in, Rs_out, lmax, size, fpix):
    return nn.Sequential(
        Convolution(Rs_in, Rs_out, lmax=lmax, size=size, stride=1, padding=size//2, fuzzy_pixels=fpix),
        #BatchNorm(Rs_out, normalization='component'),
        NormActivation(Rs_out, swish, normalization = 'component'),
    )

class Encode(nn.Module):
    def __init__(self, size, mult, lmax, inp_channels=11):
        super(Encode, self).__init__()
        
        m = mult
        Rs_in = [(inp_channels,0)]
        Rs = list(range(lmax + 1)) #base Rs
        fp = False  #option to add noise to conv kernels
        
        # Down sampling
        self.down0 = ConvBlock(Rs_in      , Rs * (m+2),  lmax=lmax, size=size, fpix=fp)
        self.down1 = ConvBlock(Rs * (m+2) , Rs *  m,     lmax=lmax, size=size, fpix=fp)
        self.down2 = ConvBlock(Rs * m     , Rs * (m//2), lmax=lmax, size=size, fpix=fp)
        self.down3 = ConvBlock(Rs * (m//2), Rs * (m//4), lmax=lmax, size=size, fpix=fp)
        #self.down4 = ConvBlock(Rs * (m//4), Rs * (m//8), lmax=lmax, size=size, fpix=fp)
        self.down4 = ConvBlock(Rs * (m//4), [(1,0)],     lmax=lmax, size=size, fpix=fp)
 
    def forward(self, x):
        #Down sampling
        down0 = self.down0(x) 
        down1 = self.down1(down0) 
        down2 = self.down2(down1) 
        down3 = self.down3(down2) 
        down4 = self.down4(down3) 
        return down4

    
class Decode(nn.Module):
    def __init__(self, size, mult, lmax, out_channels=11):
        super(Decode, self).__init__()
        
        m = mult
        Rs_out =  [(out_channels,0)]
        m = mult #multiplier
        Rs = list(range(lmax + 1)) #base Rs
        fp = False  #option to add noise to conv kernels

        # Up sampling
        #self.up1 = ConvBlock(Rs * (m//8), Rs*(m//4), lmax=lmax, size=size, fpix=fp)
        self.up0 = ConvBlock([(1,0)],   Rs*(m//4), lmax=lmax, size=size, fpix=fp)
        self.up1 = ConvBlock(Rs*(m//4), Rs*(m//2), lmax=lmax, size=size, fpix=fp)
        self.up2 = ConvBlock(Rs*(m//2), Rs*m     , lmax=lmax, size=size, fpix=fp)
        self.up3 = ConvBlock(Rs*m     , Rs*(m+2) , lmax=lmax, size=size, fpix=fp)
        self.up4 = ConvBlock(Rs*(m+2) , Rs_out   , lmax=lmax, size=size, fpix=fp)
        
          
    def forward(self, x):
        # Up sampling
        up0 = self.up0(x) 
        up1 = self.up1(up0) 
        up2 = self.up2(up1) 
        up3 = self.up3(up2)
        up4 = self.up4(up3)
        
        return up4

    
if __name__ == "__main__":

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  inp_size = 16
  inp_channels = 11
  out_channels = 11

  lmax = 0
  k_size = 3
  m = 8 #multiplier
  
  x = torch.Tensor(1, inp_size, inp_size, inp_size, inp_channels)
  x.to(device)
  print("x size: {}".format(x.size()))
  
  modelEncode = Encode(size=k_size, mult=m, lmax=lmax, inp_channels=inp_channels)
  out1 = modelEncode(x)
  print("Encoded out size: {}".format(out1.size()))

  modelDecode = Decode(size=k_size, mult=m, lmax=lmax, out_channels=out_channels)
  out2 = modelDecode(out1)
  print("Decode out size: {}".format(out2.size()))
