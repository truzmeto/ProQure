"""
SE(3) Unet model by T. Ruzmetov
"""

import sys
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from e3nn.non_linearities.rescaled_act import sigmoid, swish, tanh
from e3nn.non_linearities.norm_activation import NormActivation
from e3nn.batchnorm import BatchNorm

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from src.SE3Model.convolution import Convolution

   
def ConvBlock(Rs_in, Rs_out, lmax, size, fpix, stride):
    return nn.Sequential(
        Convolution(Rs_in, Rs_out, lmax=lmax,  size=size, stride=stride, padding = size//2, bias=None,fuzzy_pixels = fpix),
        NormActivation(Rs_out, swish, normalization = 'component'),
    )


def ConvTransBlock(Rs_in, Rs_out, lmax, size, fpix):
    return nn.Sequential(
        Convolution(Rs_in, Rs_out, lmax=lmax, size=size, stride=2, padding=size//2, bias=None, output_padding=1, transpose=True, fuzzy_pixels = fpix),
        NormActivation(Rs_out, swish, normalization = 'componenet'),
    )


class Encode(nn.Module):
    def __init__(self, size, lmax, inp_channels=11, n_aa=3):
        super(Encode, self).__init__()
        
        Rs_in = [(inp_channels,0)]
        Rs = list(range(lmax + 1)) #base Rs
        fp = False  #option to add noise to conv kernels
        st = 2      #downlampling stride

        self.down_0 = ConvBlock(Rs_in,    10   * Rs, lmax=lmax, size=size, fpix=fp, stride=st)
        self.down_1 = ConvBlock(10 *  Rs, 8    * Rs, lmax=lmax, size=size, fpix=fp, stride=st)
        self.down_2 = ConvBlock(8 *   Rs, 6    * Rs, lmax=lmax, size=size, fpix=fp, stride=st)
        self.down_3 = ConvBlock(6 *   Rs, 4    * Rs, lmax=lmax, size=size, fpix=fp, stride=st)
        self.down_4 = ConvBlock(4 *   Rs, n_aa * Rs, lmax=lmax, size=size, fpix=fp, stride=st) 
        

    def forward(self, x):
        # Down sampling
        down_0 = self.down_0(x)
        down_1 = self.down_1(down_0)
        down_2 = self.down_2(down_1) 
        down_3 = self.down_3(down_2) 
        down_4 = self.down_4(down_3) 
        return down_4

    

class Decode(nn.Module):
    def __init__(self, size, lmax, out_channels=11, n_aa=3):
        super(Decode, self).__init__()
        
        Rs_out =  [(out_channels,0)]
        Rs = list(range(lmax + 1)) #base Rs
        fp = False  #option to add noise to conv kernels

        # Up sampling
        self.trans_0 = ConvTransBlock(n_aa * Rs, n_aa * Rs, lmax=lmax, size=size, fpix=fp)
        self.up_0    = ConvBlock( n_aa * Rs, 4 * Rs, lmax=lmax, size=size, fpix=fp, stride=1)

        self.trans_1 = ConvTransBlock(4 * Rs, 4 * Rs, lmax=lmax, size=size, fpix=fp )
        self.up_1    = ConvBlock(    4 * Rs, 6 * Rs, lmax=lmax, size=size, fpix=fp, stride=1)

        self.trans_2 = ConvTransBlock(6 * Rs, 6 * Rs, lmax=lmax, size=size, fpix=fp)
        self.up_2    = ConvBlock(    6 * Rs, 8 * Rs, lmax=lmax, size=size, fpix=fp, stride=1)

        self.trans_3 = ConvTransBlock(8 * Rs,  8 * Rs, lmax=lmax, size=size, fpix=fp)
        self.up_3    = ConvBlock(    8 * Rs, 10 * Rs, lmax=lmax, size=size, fpix=fp, stride=1)

        self.trans_4 = ConvTransBlock(10* Rs, 10 * Rs, lmax=lmax, size=size, fpix=fp)
        self.up_4    = ConvBlock(    10* Rs, Rs_out, lmax=lmax, size=size, fpix=fp, stride=1)
    
        
    def forward(self, x):
        
        # Up sampling
        trans_0 = self.trans_0(x) 
        up_0 = self.up_0(trans_0) 
                       
        trans_1 = self.trans_1(up_0) 
        up_1 = self.up_1(trans_1)
       
        trans_2 = self.trans_2(up_1) 
        up_2 = self.up_2(trans_2)  
        
        trans_3 = self.trans_3(up_2) 
        up_3 = self.up_3(trans_3) 
              
        trans_4 = self.trans_4(up_3) 
        up_4 = self.up_4(trans_4) 
        
           
        return up_4

    
if __name__ == "__main__":
     
    inp_size = 32
    inp_channels = 11
    out_channels = 11

    lmax = 1
    k_size = 3
   
    x = torch.Tensor(1, inp_size, inp_size, inp_size, inp_channels).cuda()
    print("x size: {}".format(x.size()))
  
    modelEncode = Encode(size=k_size, lmax=lmax, inp_channels=inp_channels).cuda()
    out1 = modelEncode(x)
    print("Encoded out size: {}".format(out1.size()))
    
  
    modelDecode = Decode(size=k_size, lmax=lmax, out_channels=inp_channels).cuda()
    out2 = modelDecode(out1)
    print("Decode out size: {}".format(out2.size()))
