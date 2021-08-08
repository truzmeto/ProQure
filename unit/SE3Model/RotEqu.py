"""
make a movie to show equivariance!
"""

import os
import sys
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import pyvista as pv
from scipy import ndimage

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from unit.Util.Shapes3D import get3D_rod
from src.SE3Model.EncDecSE3 import Encode, Decode

def rotate_ligand(ligand, rotation_angle):    
    
    ligand = ndimage.interpolation.rotate(ligand,
                                          angle = rotation_angle,
                                          axes=(2,3),
                                          reshape=False,
                                          order=1,
                                          mode= 'nearest',#'constant',
                                          cval=0.0)
    
    return ligand



########---------- Simple test with 1 forward pass -----------########

b, c, d, h, w = 1, 11, 32, 32, 32
dim = (b, c, d, h, w)
dev_id = 0
torch.manual_seed(1000)


inp_channels = 11
lmax = 2
k_size = 3
m = 8 #multiplier


torch.cuda.set_device(dev_id)
modelDecode = Decode(size=k_size, mult=m, lmax=lmax, out_channels=inp_channels).cuda()
modelEncode = Encode(size=k_size, mult=m, lmax=lmax, inp_channels=inp_channels).cuda()


#plot output density maps
chan_id = 0 # can be 0,1,2,3
fs = 16; cmap = 'gist_ncar'#'rainbow'
inp =  get3D_rod(dim).cuda()

# Open a movie file
p = pv.Plotter(point_smoothing = True, shape=(1, 3))
p.open_movie('movieRotEqu.mp4')



for i in range(2):

    inpR = rotate_ligand(inp.cpu().detach().numpy(), rotation_angle= i*2.0)
    inpR = torch.from_numpy(inpR).float().to('cuda')
    
    dataR = torch.einsum('tixyz->txyzi', inpR)
    modelEncode.eval()
    latentR = modelEncode(dataR)

    modelDecode.eval()
    outputR = modelDecode(latentR)
    outputR = torch.einsum('txyzi->tixyz', outputR) #unpermute
    latentR = torch.einsum('txyzi->tixyz', latentR) #unpermute
    
    if i == 0:
        output = outputR
        norm = output.max() - output.min()
        
    outputUR = rotate_ligand(outputR.cpu().detach().numpy(), rotation_angle= -i*2.0)
    outputUR = torch.from_numpy(outputUR).float().to('cuda')
    err = (output - outputUR).pow(2).mean().sqrt() # / norm
    err = err.item()
    
    p.subplot(0, 0)
    chan1 = inpR[0,chan_id,:,:,:].cpu().detach().numpy()
    p.add_text("Input ", position = 'lower_left', font_size = fs)
    p.add_text("RMSE =  " + str(round(err,5)), position = 'upper_left', font_size = fs-2)
    p.add_volume(np.abs(chan1), cmap = cmap, opacity = "linear", show_scalar_bar=False)
    
    p.subplot(0, 1)
    chan2 = latentR[0,chan_id,:,:,:].cpu().detach().numpy()
    p.add_text("Latent ", position = 'lower_left', font_size = fs)
    p.add_volume(np.abs(chan2), cmap = cmap, opacity = "linear", show_scalar_bar=False)
    

    p.subplot(0, 2)
    chan3 = outputR[0,chan_id,:,:,:].cpu().detach().numpy()
    p.add_text("Output", position = 'lower_left', font_size = fs)
    p.add_volume(np.abs(chan3), cmap = cmap, opacity = "linear", show_scalar_bar=False)

    if i == 0 :
        p.show(auto_close=False)

    p.write_frame()
    p.clear()

p.close()
