import torch
import torch.nn as nn
import torch.nn.functional as F


def ConvBlock(in_dim, out_dim, size, act):
    return nn.Sequential(
        nn.Conv3d(in_dim, out_dim, kernel_size=size, stride=1, padding=size//2, bias=False),
        nn.BatchNorm3d(out_dim),
        act,
    )

class Encode(nn.Module):
    def __init__(self, in_dim, size, mult):
        super(Encode, self).__init__()
        
        m = mult
        act = nn.ReLU()
       
        # Down sampling
        self.down_0 = ConvBlock(in_dim, m+2,  size, act)
        self.down_1 = ConvBlock(m+2,    m,    size, act)
        self.down_2 = ConvBlock(m,      m//2, size, act)
        self.down_3 = ConvBlock(m//2,   m//4, size, act)
        self.down_4 = ConvBlock(m//4,   m//8, size, act) 

  
    def forward(self, x):
        # Down sampling
        down_0 = self.down_0(x) 
        down_1 = self.down_1(down_0) 
        down_2 = self.down_2(down_1) 
        down_3 = self.down_3(down_2) 
        down_4 = self.down_4(down_3) 
       
        return down_4


    
class Decode(nn.Module):
    def __init__(self, out_dim, size, mult):
        super(Decode, self).__init__()
        
        m = mult
        act = nn.ReLU()
        
        # Up sampling
        self.up_0 = ConvBlock(m//8, m//4,    size, act)
        self.up_1 = ConvBlock(m//4, m//2,    size, act)
        self.up_2 = ConvBlock(m//2, m,       size, act)
        self.up_3 = ConvBlock(m   , m+2,     size, act) 
        self.up_4 = ConvBlock(m+2 , out_dim, size, act) 

  
        
    def forward(self, x):
        
        # Up sampling
        up_0 = self.up_0(x) 
        up_1 = self.up_1(up_0) 
        up_2 = self.up_2(up_1) 
        up_3 = self.up_3(up_2)
        up_4 = self.up_4(up_3)
        

        return up_4

    
if __name__ == "__main__":
    
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  inp_size = 16
  in_dim = 11
  out_dim = 11
  
  x = torch.Tensor(1, in_dim, inp_size, inp_size, inp_size)
  x.to(device)
  print("x size: {}".format(x.size()))
  
  modelEncode = Encode(in_dim = in_dim,size = 5, mult = 8)
  out1 = modelEncode(x)
  print("Encoded out size: {}".format(out1.size()))

  
  modelDecode = Decode(out_dim = out_dim, size = 5, mult = 8)
  out2 = modelDecode(out1)
  
  print("Decode out size: {}".format(out2.size()))
