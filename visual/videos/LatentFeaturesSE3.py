import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import matplotlib.pyplot as plt
import pyvista as pv
from scipy import ndimage

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from src.Util.volume import get_volume
from src.SE3Model.EncDecSE3 import Encode


def get_inp(pdb_ids, pdb_path, dim, rotate = True):
    """
    This function takes care of all inputs: 
    input volume <--> output volume, + rotations etc..
    """
    norm = False
    resolution = 1.000
    box_size = int(dim*resolution)
    batch_list = [pdb_path + str(ids) + ".pdb" for ids in pdb_ids]
    with torch.no_grad():
        inp, _, _ = get_volume(path_list = batch_list, 
                               box_size = box_size,
                               resolution = resolution,
                               norm = norm,
                               rot = False,
                               trans = False)
    return inp 


if __name__=='__main__':
        
    dev_id = 0    
    batch_size = 1
    dim = 20
    pdb_ids = ["AYA"]
    tp_name = pdb_ids[0]
    
    pdb_path  = "/u1/home/tr443/Projects/ProteinQure/data/Trajectories/" + tp_name + "/" + tp_name
    out_path = '/u1/home/tr443/Projects/ProteinQure/ProQure/output/'
    params_file_name = str(100000) + 'net_paramsSE'
    inp_channels = 11
    lmax = 0
    k_size = 3
    m = 8 #multiplier
   
    torch.cuda.set_device(dev_id)
    modelEncode = Encode(size=k_size, mult=m, lmax=lmax, inp_channels=inp_channels).cuda()
    checkpoint = torch.load(out_path + params_file_name)
    modelEncode.load_state_dict(checkpoint['modelEncode'])


    start = 1001    
    end = start + 200

    fs = 12; cmap = "viridis_r" #'gist_ncar'
    batch_id = 0
    p = pv.Plotter(point_smoothing = True)
    p.open_movie('my_movie.mp4')
    
    for i in range(start, end, 1):
        volume = get_inp([i], pdb_path, dim, rotate = False)
        volume = torch.einsum('tixyz->txyzi', volume) #permute

        modelEncode.eval()
        latent = modelEncode(volume)
                
        out = latent.squeeze().cpu().detach().numpy() 
        text = tp_name + str(i)   
        p.add_text(text, position = 'upper_left', font_size = fs)
        p.add_volume(np.abs(out), cmap = cmap, clim=[0.0, 12.0],opacity = "linear")
        p.add_axes()

        if i == start :
            p.show(auto_close=False)
      
        
        p.write_frame()
        p.clear()
        
    p.close()
    #p.show()
    
    
  
       
