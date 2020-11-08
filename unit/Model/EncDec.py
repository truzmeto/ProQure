import os
import sys
import torch
import numpy as np
import pyvista as pv

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from src.Model.EncDec1 import EDNet
from unit.Util.Shapes3D import get3D_rod

########---------- Simple test with 1 forward pass -----------########

size = 32
b, c, d, h, w = 1, 1, size, size, size
dim = (b, c, d, h, w)
data =  get3D_rod(dim).cuda()
torch.manual_seed(1000)

in_dim = c; out_dim = c
model = EDNet(in_dim = in_dim, out_dim = out_dim, size = 5, mult = 8).cuda()
output = model(data)

#print("11 input channels VS 6 output channels")
print("Input dimension -->", data.size())
print("Output dimension -->", output.size())


#plot output density map
chan_id = 0 # can be 0,1,2,3
channel = data[0,chan_id,:,:,:].cpu().detach().numpy()
fs = 16; cmap = 'gist_ncar'#'rainbow'
p = pv.Plotter(point_smoothing = True, shape=(1, 2))

p.subplot(0, 0)
p.add_text("Input", position = 'upper_left', font_size = fs)
p.add_volume(np.abs(channel), cmap = cmap, opacity = "linear")

channel = output[0,chan_id,:,:,:].cpu().detach().numpy()
p.subplot(0, 1)
p.add_text("Output", position = 'upper_left', font_size = fs)
p.add_volume(np.abs(channel), cmap = cmap, opacity = "linear")

p.show()


