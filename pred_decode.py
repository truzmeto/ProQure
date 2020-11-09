import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import matplotlib.pyplot as plt
import pyvista as pv

from src.Util.volume import get_volume
from src.Model.EncDec2 import Decode

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
        inp, _ = get_volume(path_list = batch_list, 
                            box_size = box_size,
                            resolution = resolution,
                            norm = norm,
                            rot = False,
                            trans = False)
    return inp 


if __name__=='__main__':


    dev_id = 0    
    batch_size = 4
    dim = 16
    start = 600
    end = start + batch_size
    test_list = list(range(start, end))
    pdb_ids = ["AAA"]
    tp_name = pdb_ids[0]

    pdb_path  = "/u1/home/tr443/Projects/ProteinQure/data/Trajectories/" + tp_name + "/" + tp_name
    out_path = 'output/'
    params_file_name = str(20000) + 'net_params'

    torch.cuda.set_device(dev_id)
    modelDecode = Decode(out_dim = 11, size = 3, mult = 8).cuda()
    checkpoint = torch.load(out_path + params_file_name)
    modelDecode.load_state_dict(checkpoint['modelDecode'])
  
    volume = get_inp(test_list, pdb_path, dim, rotate = False)
    latent = torch.load(out_path + "/" + tp_name + "/" + "latent" + str(start) + "-" + str(end) + ".pt")

    modelDecode.eval()
    output = modelDecode(latent)
    
    err = (output - volume).pow(2).mean().sqrt().item()
    err = round(err,3)
    print("RMSE", err)

    pl = pv.Plotter(point_smoothing = True, shape=(1, 2))
    fs = 12
    batch_id = 0

    #vol = volume.sum(dim=1)[batch_id].detach().cpu().numpy()
    #out = output.sum(dim=1)[batch_id].detach().cpu().numpy()

    #text = tp_name + str(batch_id + 1)
    #print(out.min())
    #print(out.max())
    #pl.subplot(0, 0);  
    #pl.add_text(text+" input", position = 'upper_left', font_size = fs)
    #pl.add_volume(vol, cmap = "viridis_r", opacity = "linear")
    #pl.subplot(0, 1); 
    #pl.add_text(text+" output", position = 'upper_left', font_size = fs)
    #pl.add_text(str(err), position = 'upper_right', font_size = fs)
    #pl.add_volume(abs(out), cmap = "viridis_r", opacity = "linear")
    #pl.add_axes()
    #pl.show()


    Agroup_names = ["Sulfur/Selenium"  , "Nitrogen Amide",
                    "Nitrogen Aromatic", "Nitrogen Guanidinium",
                    "Nitrogen Ammonium", "Oxygen Carbonyl",
                    "Oxygen Hydroxyl"  , "Oxygen Carboxyl",
                    "Carbon sp2"       , "Carbon Aromatic",
                    "Carbon sp3"]

    vol = volume.detach().cpu().numpy()
    out = output.detach().cpu().numpy()

    
    pl = pv.Plotter(point_smoothing = True, shape=(2, 6))
    fs = 12
    i_group = 0
    ipdb = 0
    
    for i in range(2):
        for j in range(6):
            
            if i_group < 11:
                ch = out[0,i_group,:,:,:]#.cpu().numpy()
                text = tp_name + ": " + Agroup_names[i_group]  
                
                pl.subplot(i, j)
                pl.add_text(text, position = 'upper_left', font_size = fs)
                pl.add_volume(ch, cmap = "viridis_r", opacity = "linear")
                
            i_group = i_group + 1

    pl.add_axes()
    pl.show()

    
    
       