import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from e3nn.non_linearities.rescaled_act import sigmoid, swish, tanh
from e3nn.non_linearities.norm_activation import NormActivation

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from src.SE3Model.convolution import Convolution

def ConvBlock(Rs_in, Rs_out, stride, size, fpix):
    return nn.Sequential(
        Convolution(Rs_in, Rs_out, size=size, stride=stride, padding=size//2, fuzzy_pixels=fpix),
        NormActivation(Rs_out, swish, normalization = 'component'),
    )

def ConvTBlock(Rs_in, Rs_out, size, fpix):
    return nn.Sequential(
        Convolution(Rs_in, Rs_out, size=size, stride=2, padding=size//2, bias=None, output_padding=1, transpose=True, fuzzy_pixels = fpix),
        #NormActivation(Rs_out, swish, normalization = 'componenet'),
    )

class Encode(nn.Module):
    def __init__(self, size, n_reps, inp_channels=4, fp=False):
        super(Encode, self).__init__()
        
        m = inp_channels
        Rs_in = [(inp_channels,0)]
        n_vec, n_ten = n_reps
        RsSca = [0]; RsVec = [1] * n_vec; RsTen = [2] * n_ten
        Rs = RsSca 
        RsH = RsVec + RsTen
     
        #Down sampling
        self.down1 = ConvBlock(Rs_in                     , Rs*m + RsH*m          , stride=1, size=size, fpix=fp)
        self.down2 = ConvBlock(Rs*m + RsH*m              , Rs*(m//2) + RsH*(m//2), stride=2, size=size, fpix=fp)
        self.down3 = ConvBlock(Rs*(m//2) + RsH*(m//2)    , Rs*(m//4)             , stride=1, size=size, fpix=fp)
       
    def forward(self, x):
        #Down sampling
        down1 = self.down1(x) 
        down2 = self.down2(down1) 
        down3 = self.down3(down2) 
        return down3

    
class Decode(nn.Module):
    def __init__(self, size, n_reps, out_channels=26, fp=False):
        super(Decode, self).__init__()
        
        Rs_out =  [(out_channels,0)]
        n_vec, n_ten = n_reps
        RsSca = [0]; RsVec = [1] * n_vec; RsTen = [2] * n_ten
        Rs = RsSca 
        RsH = RsVec + RsTen
        
        # Up sampling
        self.up1 = ConvBlock(Rs          , Rs*2 + RsH*2, stride=1, size=size, fpix=fp)
        self.up2 = ConvBlock(Rs*2 + RsH*2, Rs*4 + RsH*4, stride=1, size=size, fpix=fp)
        self.up3 = ConvBlock(Rs*4 + RsH*4, Rs*8 + RsH*8, stride=1, size=size, fpix=fp)
        
        self.tu4 = ConvTBlock(Rs*8  + RsH*8 , Rs*8  + RsH*8 , size=size, fpix=fp)
        self.up4 = ConvBlock( Rs*8  + RsH*8 , Rs*16 + RsH*16, stride=1 , size=size, fpix=fp)
        self.up5 = ConvBlock( Rs*16 + RsH*16, Rs_out        , stride=1 , size=size, fpix=fp)
                  
    def forward(self, x):
        # Up sampling
        up1 = self.up1(x) 
        up2 = self.up2(up1) 
        up3 = self.up3(up2) 
        
        tu4 = self.tu4(up3)
        up4 = self.up4(tu4) 
        up5 = self.up5(up4) 
        return up5

    
if __name__ == "__main__":

    import warnings
    warnings.filterwarnings("ignore")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    inp_size = 20
    inp_channels = 4
    out_channels = 26
    n_reps = [1,1]
    k_size = 5
    
    x = torch.rand(1, inp_size, inp_size, inp_size, inp_channels)
    x.to(device)
    print("Input size: {}".format(x.size()))
    
    modelEncode = Encode(size=k_size, n_reps = n_reps, inp_channels=inp_channels)
    out1 = modelEncode(x)
    print("Latent out size: {}".format(out1.size()))
    
    modelDecode = Decode(size=k_size, n_reps = n_reps, out_channels=out_channels)
    out2 = modelDecode(out1)
    print("Out size: {}".format(out2.size()))
    print("Out max val: {}".format(out2.max()))
    
