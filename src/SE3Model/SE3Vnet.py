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


def ConvBlock1(Rs_in, Rs_out, size, fpix):
    return nn.Sequential(
        Convolution(Rs_in, Rs_out, size = size, stride = 1, padding = size//2, fuzzy_pixels = fpix),
        NormActivation(Rs_out, swish, normalization = 'component'),
    )

   
def ConvBlock2(Rs_in, Rs_out, size, fpix, stride):
    return nn.Sequential(
        ConvBlock1(Rs_in, Rs_out, size, fpix),
        Convolution(Rs_out, Rs_out, size=size, stride=stride, padding = size//2, fuzzy_pixels = fpix),
    )


def ConvTransBlock(Rs_in, Rs_out, size, fpix):
    return nn.Sequential(
        Convolution(Rs_in, Rs_out, size=size, stride=2, padding=size//2, output_padding=1, transpose=True, fuzzy_pixels = fpix),
        #NormActivation(Rs_out, swish, normalization = 'componenet'),
    )



class VNet(nn.Module):
    def __init__(self, size, n_reps, inp_channels=11):
        super(VNet, self).__init__()
        
        Rs_in = [(inp_channels,0)]
        Rs_out =  [(inp_channels,0)]
        m = 10
        n_vec, n_ten = n_reps
        RsSca = [0]; RsVec = [1] * n_vec; RsTen = [2] * n_ten
        Rs = RsSca * m
        Rs1 = RsVec + RsTen
        
        fp = False  #option to add noise to conv kernels
        st = 2      #downlampling stride
        
        #Down sampling
        self.down_1 = ConvBlock2(Rs_in,              Rs + Rs1, size=size, fpix=fp, stride=st)
        self.down_2 = ConvBlock2(     Rs + Rs1, 2  * Rs + Rs1, size=size, fpix=fp, stride=st)
        self.down_3 = ConvBlock2( 2 * Rs + Rs1, 4  * Rs + Rs1, size=size, fpix=fp, stride=st)
        self.down_4 = ConvBlock2( 4 * Rs + Rs1, 8  * Rs + Rs1, size=size, fpix=fp, stride=st) 
        self.down_5 = ConvBlock2( 8 * Rs + Rs1, 16 * Rs + Rs1, size=size, fpix=fp, stride=st)
        
        # Bridge
        self.bridge = ConvBlock2(16 * Rs + Rs1, 32 * Rs + Rs1, size=size, fpix=fp, stride=1)        
                
        # Up sampling
        self.trans_1 = ConvTransBlock( 32 * Rs +   Rs1, 32 * Rs + Rs1, size=size, fpix=fp)
        self.up_1    = ConvBlock2(     48 * Rs + 2*Rs1, 16 * Rs + Rs1, size=size, fpix=fp, stride=1)
        self.trans_2 = ConvTransBlock( 16 * Rs +   Rs1, 16 * Rs + Rs1, size=size, fpix=fp )
        self.up_2    = ConvBlock2(     24 * Rs + 2*Rs1, 8  * Rs + Rs1, size=size, fpix=fp, stride=1)
        self.trans_3 = ConvTransBlock( 8  * Rs +   Rs1, 8  * Rs + Rs1, size=size, fpix=fp)
        self.up_3    = ConvBlock2(     12 * Rs + 2*Rs1, 4  * Rs + Rs1, size=size, fpix=fp, stride=1)
        self.trans_4 = ConvTransBlock( 4  * Rs +   Rs1, 4  * Rs + Rs1, size=size, fpix=fp)
        self.up_4    = ConvBlock2(     6  * Rs + 2*Rs1, 2  * Rs + Rs1, size=size, fpix=fp, stride=1)
        self.trans_5 = ConvTransBlock( 2  * Rs +   Rs1, 2  * Rs + Rs1, size=size, fpix=fp)
        self.up_5    = ConvBlock2(     3  * Rs + 2*Rs1, 1  * Rs + Rs1, size=size, fpix=fp, stride=1)
                                                                     
        # Output
        self.out = ConvBlock1(Rs + Rs1, Rs_out, size=size, fpix=fp)

    
    def skip(self, uped, bypass):
        """
        This functions matches size between bypass and upsampled feature.
        It pads or unpads bypass depending on how different dims are.
        """

        uped = torch.einsum('txyzi->tixyz', uped)
        bypass = torch.einsum('txyzi->tixyz', bypass)
        p = bypass.shape[2] - uped.shape[2]                                                                                          
    
        if p == 0:
            out = torch.cat((uped, bypass), 1)
            out = torch.einsum('tixyz->txyzi', out)
            return out
        else:
            pl = p // 2
            pr = p - pl
            bypass = F.pad(bypass, (-pl, -pr, -pl, -pr, -pl, -pr))
            out = torch.cat((uped, bypass), 1)
            out = torch.einsum('tixyz->txyzi', out)
            
            return out
    
        
    def forward(self, x):
        # Down sampling
        down_1 = self.down_1(x) 
        down_2 = self.down_2(down_1) 
        down_3 = self.down_3(down_2) 
        down_4 = self.down_4(down_3) 
        down_5 = self.down_5(down_4) 
        bridge = self.bridge(down_5) 

        # Up sampling
        trans_1 = self.trans_1(bridge) 
        concat_1 = self.skip(trans_1, down_5)
        up_1 = self.up_1(concat_1) 
                       
        trans_2 = self.trans_2(up_1) 
        concat_2 = self.skip(trans_2, down_4)
        up_2 = self.up_2(concat_2)
        
        trans_3 = self.trans_3(up_2) 
        concat_3 = self.skip(trans_3, down_3)
        up_3 = self.up_3(concat_3)  
        
        trans_4 = self.trans_4(up_3) 
        concat_4 = self.skip(trans_4, down_2)
        up_4 = self.up_4(concat_4) 
        
        trans_5 = self.trans_5(up_4) 
        concat_5 = self.skip(trans_5, down_1)
        up_5 = self.up_5(concat_5) 

        # Output
        out = self.out(up_5)
        
        return out

    
if __name__ == "__main__":
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  inp_size = 32
  inp_channels = 11
  n_reps = [3,3] # n_vectors (L=1), n_tensors (L=2)
  
  k_size = 3
  
  x = torch.Tensor(1, inp_size, inp_size, inp_size, inp_channels)
  x.to(device)
  print("Input size: {}".format(x.size()))
  
  model = VNet(size = k_size, n_reps = n_reps)
  
  out = model(x)
  #print("latent size: {}".format(latent.size()))
  print("output size: {}".format(out.size()))
  
