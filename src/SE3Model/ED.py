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

def ConvBlock(Rs_in, Rs_out, size, stride, fpix):
    return nn.Sequential(
        Convolution(Rs_in, Rs_out, size = size, stride = stride, padding = size//2, bias=None, fuzzy_pixels = fpix),
        NormActivation(Rs_out, swish, normalization = 'component'),
    )


def ConvTBlock(Rs_in, Rs_out, size, fpix):
    return nn.Sequential(
        Convolution(Rs_in, Rs_out, size=size, stride=2, padding=size//2, bias=None,output_padding=1, transpose=True, fuzzy_pixels = fpix),
        #NormActivation(Rs_out, swish, normalization = 'componenet'),
    )


class VNet(nn.Module):
    def __init__(self, size, n_reps, inp_channels=11):
        super(VNet, self).__init__()
        
        Rs_in = [(inp_channels,0)]
        Rs_out =  [(inp_channels,0)]
        m = 11
        n_vec, n_ten = n_reps
        RsSca = [0]; RsVec = [1] * n_vec; RsTen = [2] * n_ten
        Rs = RsSca * m
        Rs1 = RsVec + RsTen
        fp = False  #option to add noise to conv kernels

        #down
        self.d1 = ConvBlock(  Rs_in,        Rs +     Rs1, size=size, stride = 2, fpix=fp)
        self.d2 = ConvBlock(  Rs + Rs1,   2*Rs + 2 * Rs1, size=size, stride = 2, fpix=fp)
        self.d3 = ConvBlock(2*Rs + 2*Rs1, 4*Rs + 4 * Rs1, size=size, stride = 2, fpix=fp)
        self.d4 = ConvBlock(4*Rs + 4*Rs1, 8*Rs + 8 * Rs1, size=size, stride = 2, fpix=fp)
        self.d5 = ConvBlock(8*Rs + 8*Rs1,16*Rs + 16* Rs1, size=size, stride = 2, fpix=fp)

        
        #up
        self.u1 = ConvTBlock(16*Rs + 16*Rs1,16*Rs + 16*Rs1, size=size, fpix=fp)
        self.cd1 = ConvBlock(16*Rs + 16*Rs1,8*Rs + 8*Rs1, size=size, stride = 1, fpix=fp)
        self.u2 = ConvTBlock(8*Rs + 8*Rs1,  8*Rs + 8*Rs1, size=size, fpix=fp)
        self.cd2 = ConvBlock(8*Rs + 8*Rs1,  4*Rs +  4*Rs1, size=size, stride = 1, fpix=fp)
        self.u3 = ConvTBlock(  4*Rs +   4*Rs1,  4*Rs +  4*Rs1, size=size, fpix=fp)
        self.cd3 = ConvBlock(4*Rs + 4*Rs1,  2*Rs +  2*Rs1, size=size, stride = 1, fpix=fp)
        self.u4 = ConvTBlock(  2*Rs +   2*Rs1,  2*Rs +  2*Rs1, size=size, fpix=fp)
        self.cd4 = ConvBlock(2*Rs + 2*Rs1,  Rs +  Rs1, size=size, stride = 1, fpix=fp)
        self.u5 = ConvTBlock(  Rs +   Rs1,  Rs +  Rs1, size=size, fpix=fp)
        self.cd5 = ConvBlock(  Rs +   Rs1,  Rs_out,     size=size, stride = 1, fpix=fp)

        
    def forward(self, x):
        # Down sampling
        d1 = self.d1(x) 
        #print("down1 shape", d1.shape)
        d2 = self.d2(d1) 
        #print("down2 shape", d2.shape)
        d3 = self.d3(d2) 
        #print("down3 shape", d3.shape)
        d4 = self.d4(d3) 
        d5 = self.d5(d4) 
       
        
        u1 = self.u1(d5) 
        cd1 = self.cd1(u1)
        #print("up1 shape", cd1.shape)

        u2 = self.u2(cd1) 
        cd2 = self.cd2(u2) 
        #print("up2 shape", cd2.shape)

        u3 = self.u3(cd2) 
        cd3 = self.cd3(u3) 
        #print("up3 shape", cd3.shape)

        u4 = self.u4(cd3) 
        cd4 = self.cd4(u4)

        u5 = self.u5(cd4) 
        cd5 = self.cd5(u5) 

        
        return cd5

    
if __name__ == "__main__":
  #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  inp_size = 32
  inp_channels = 11
  n_reps = [3,3] # n_vectors (L=1), n_tensors (L=2)
  
  k_size = 3
  
  x = torch.Tensor(1, inp_size, inp_size, inp_size, inp_channels).cuda()
  #x.to(device)
  print("Input size: {}".format(x.size()))
  
  model = VNet(size = k_size, n_reps = n_reps).cuda()
  
  out = model(x)
  #print("latent size: {}".format(latent.size()))
  print("output size: {}".format(out.size()))
  
  #target = torch.ones(1, 4, 4, 4, 64)
