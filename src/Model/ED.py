"""
SE(3) Unet model by T. Ruzmetov
"""

import sys
import os

import torch
import torch.nn as nn
import torch.nn.functional as F


def ConvBlock0(inp_dim, out_dim, size, act, stride, pool=True):
    return nn.Sequential(
        nn.Conv3d(inp_dim, out_dim, kernel_size=size, stride=stride, bias=False, padding=size//2 ),
        act,
    )

def ConvBlock(inp_dim, out_dim, size, act, stride, pool=True):
    return nn.Sequential(
        nn.Conv3d(inp_dim, out_dim, kernel_size=size, stride=stride, bias=False, padding=size//2 ),
        #nn.BatchNorm3d(out_dim),
        act,
        nn.MaxPool3d(kernel_size = size, stride = 1, padding = size//2),
    )


def ConvTBlock(inp_dim, out_dim, size, act):
    return nn.Sequential(
        nn.ConvTranspose3d(inp_dim, out_dim, kernel_size=size, stride=2, bias=False, padding=size//2, output_padding=1),
        #nn.BatchNorm3d(out_dim),
        #act,
    )


class VNet(nn.Module):
    def __init__(self, size, inp_channels=11):
        super(VNet, self).__init__()

        m = inp_channels
        act = nn.ReLU()
        inp_dim = inp_channels
        out_dim = inp_channels
        
        #down
        self.dw0 = ConvBlock(inp_dim, m * 2, size, act, stride=2)
        self.dw1 = ConvBlock(m * 2,   m * 4, size, act, stride=2)
        self.dw2 = ConvBlock(m * 4,   m * 8, size, act, stride=2)
        self.dw3 = ConvBlock(m * 8,   m * 16, size, act, stride=2)
        self.dw4 = ConvBlock(m * 16,  m * 32, size, act, stride=2)
   
        # Up sampling
        self.tr0 = ConvTBlock(m * 32, m * 32, size, act)            # increase dim   
        self.up0 = ConvBlock( m * 32, m * 16, size, act, stride=1)  # increase channel
        self.tr1 = ConvTBlock(m * 16, m * 16, size, act)
        self.up1 = ConvBlock( m * 16, m * 8, size, act, stride=1)
        self.tr2 = ConvTBlock(m * 8,  m * 8, size, act)
        self.up2 = ConvBlock( m * 8,  m * 4, size, act, stride=1)
        self.tr3 = ConvTBlock(m * 4,  m * 4, size, act)
        self.up3 = ConvBlock( m * 4,  m * 2, size, act, stride=1)
        self.tr4 = ConvTBlock(m * 2,  m * 2, size, act)
        self.up4 = ConvBlock0(m * 2,  out_dim, size, act, stride=1)

        
        
    def forward(self, x):
        # Down sampling
        dw0 = self.dw0(x) 
        dw1 = self.dw1(dw0) 
        dw2 = self.dw2(dw1) 
        dw3 = self.dw3(dw2) 
        dw4 = self.dw4(dw3) 
        #print(dw4.shape)
        
        tr0 = self.tr0(dw4) 
        up0 = self.up0(tr0)
        tr1 = self.tr1(up0) 
        up1 = self.up1(tr1)
        tr2 = self.tr2(up1) 
        up2 = self.up2(tr2)
        tr3 = self.tr3(up2) 
        up3 = self.up3(tr3)
        tr4 = self.tr4(up3) 
        up4 = self.up4(tr4)
        
        return up4

    
if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    inp_size = 32
    in_dim = 11
    out_dim = 11
    
    x = torch.Tensor(1, in_dim, inp_size, inp_size, inp_size)
    x.to(device)
    print("x size: {}".format(x.size()))
    
    model = VNet(size = 5)
    out = model(x)
    print("out size: {}".format(out.size()))
