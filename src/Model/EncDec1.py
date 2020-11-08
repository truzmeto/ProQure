import torch
import torch.nn as nn
import torch.nn.functional as F

def ConvBlock1(in_dim, out_dim, size, act):
    return nn.Sequential(
        nn.Conv3d(in_dim, out_dim, kernel_size=size, stride=1, padding=size//2 ),
        nn.BatchNorm3d(out_dim),
        act,
    )

def ConvBlock2(in_dim, out_dim, size, act, stride):
    return nn.Sequential(
        ConvBlock1(in_dim, out_dim, size, act),
        nn.Conv3d(out_dim, out_dim, kernel_size=size, stride=stride, padding=size//2), 
        nn.BatchNorm3d(out_dim),
    )


def ConvTransBlock(in_dim, out_dim, size, act):
    return nn.Sequential(
        nn.ConvTranspose3d(in_dim, out_dim, kernel_size=size, stride=2, padding=size//2, output_padding=1),
        nn.BatchNorm3d(out_dim),
        act,
    )


class EDNet(nn.Module):
    def __init__(self, in_dim, out_dim, size, mult):
        super(EDNet, self).__init__()

        
        m = mult
        act = nn.LeakyReLU(0.2, inplace=True)
        
        # Down sampling
        self.down_1 = ConvBlock2(in_dim, m * 2, size, act, stride=2)
        self.down_2 = ConvBlock2(m * 2, m * 4, size, act, stride=2)
        self.down_3 = ConvBlock2(m * 4, m * 8, size, act, stride=2)
        #self.down_4 = ConvBlock2(m * 8, m * 16, size, act, stride=2)

        self.down_4 = nn.Sequential(
            nn.Conv3d(m*8, m*16, kernel_size=size, stride=1, padding=size//2 ),
            act,
            nn.Conv3d(m*16, m*16, kernel_size=size, stride=2, padding=size//2),
        )
        
        
        # Up sampling
        self.trans_1 = ConvTransBlock(m * 16, m * 16, size, act)    # reduce dim   
        self.up_1 = ConvBlock2(m * 16, m * 8, size, act, stride=1) # increase channel
        

        self.trans_2 = ConvTransBlock(m * 8, m * 8, size, act)
        self.up_2 = ConvBlock2(m * 8, m * 4, size, act, stride=1)

        self.trans_3 = ConvTransBlock(m * 4, m * 4, size, act)
        self.up_3 = ConvBlock2(m * 4, m * 2, size, act, stride=1)

        self.trans_4 = ConvTransBlock(m * 2, m * 2, size, act)
        self.up_4 = ConvBlock2(m * 2, m * 1, size, act, stride=1)
        
        # Output
        self.out = ConvBlock1(m , out_dim, size, act)

  
        
    def forward(self, x):
        # Down sampling
        down_1 = self.down_1(x) 
        #print("down1", down_1.shape)
        
        down_2 = self.down_2(down_1) 
        #print("down2", down_2.shape)
        
        down_3 = self.down_3(down_2) 
        #print("down3", down_3.shape) 

        down_4 = self.down_4(down_3) 
        #print("down4", down_4.shape) 
        
  
        
        # Up sampling
        trans_1 = self.trans_1(down_4) 
        up_1 = self.up_1(trans_1) 
        #print("trans1", trans_1.shape)
        #print('up1', up_1.shape)         
                       
        trans_2 = self.trans_2(up_1) 
        up_2 = self.up_2(trans_2)
        #print("trans2", trans_2.shape)  
        #print('up2', up_2.shape)        

        
        trans_3 = self.trans_3(up_2) 
        up_3 = self.up_3(trans_3)  
        #print("trans3", trans_3.shape)  
        #print('up3', up_3.shape)

        trans_4 = self.trans_4(up_3) 
        up_4 = self.up_4(trans_4)  
        #print("trans4", trans_4.shape)  
        #print('up4', up_4.shape)
        
        
        # Output
        out = self.out(up_4)
        return out

    
if __name__ == "__main__":
    
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  inp_size = 16
  in_dim = 11
  out_dim = 11
  
  x = torch.Tensor(1, in_dim, inp_size, inp_size, inp_size)
  x.to(device)
  print("x size: {}".format(x.size()))
  
  model = EDNet(in_dim = in_dim, out_dim = out_dim, size = 5, mult = 8)
  
  out = model(x)
  print("out size: {}".format(out.size()))
