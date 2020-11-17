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
from src.SE3Model.EncDecSE3 import Decode


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
        inp, _, cent = get_volume(path_list = batch_list, 
                                  box_size = box_size,
                                  resolution = resolution,
                                  norm = norm,
                                  rot = False,
                                  trans = False)
    return inp, cent 


if __name__=='__main__':


    dev_id = 0    
    batch_size = 1
    dim = 24
    start = 401
    end = start + batch_size
    test_list = list(range(start, end))
    pdb_ids = ["AYA"]
    tp_name = pdb_ids[0]

    pdb_path  = "/u1/home/tr443/Projects/ProteinQure/data/Trajectories/" + tp_name + "/" + tp_name
    out_path = 'output/'
    params_file_name = str(30000) + 'net_paramsSE'
    inp_channels = 11
    lmax = 0
    k_size = 3
    m = 8 #multiplier

    torch.cuda.set_device(dev_id)
    modelDecode = Decode(size=k_size, mult=m, lmax=lmax, out_channels=inp_channels).cuda()
    checkpoint = torch.load(out_path + params_file_name)
    modelDecode.load_state_dict(checkpoint['modelDecode'])
  
    volume, _ = get_inp(test_list, pdb_path, dim, rotate = False)
    latent = torch.load(out_path + "/" + tp_name + "/" + "latent" + str(start) + "-" + str(end-1) + ".pt")

    modelDecode.eval()
    output = modelDecode(latent)
    output = torch.einsum('txyzi->tixyz', output) #unpermute
    
    err = (output - volume).pow(2).mean().sqrt().item()
    err = round(err,3)
    print("RMSE", err)

    pl = pv.Plotter(point_smoothing = True, shape=(1, 2))
    fs = 12
    batch_id = 0
    vol = volume.sum(dim=1)[batch_id].detach().cpu().numpy()
    out = output.sum(dim=1)[batch_id].detach().cpu().numpy()
    text = tp_name + str(start + batch_id)
    pl.subplot(0, 0);  
    pl.add_text(text+" input", position = 'upper_left', font_size = fs)
    pl.add_volume(vol, cmap = "viridis_r", opacity = "linear")
    pl.subplot(0, 1); 
    pl.add_text(text+" output", position = 'upper_left', font_size = fs)
    pl.add_text("RMSE = "+str(err), position = 'upper_right', font_size = fs)
    pl.add_volume(abs(out), cmap = "viridis_r", opacity = "linear")
    pl.add_axes()
    pl.show()
    
    #Agroup_names = ["Sulfur/Selenium"  , "Nitrogen Amide",
    #                "Nitrogen Aromatic", "Nitrogen Guanidinium",
    #                "Nitrogen Ammonium", "Oxygen Carbonyl",
    #                "Oxygen Hydroxyl"  , "Oxygen Carboxyl",
    #                "Carbon sp2"       , "Carbon Aromatic",
    #                "Carbon sp3"]
    #
    ##vol = volume.detach().cpu().numpy()
    #print(output.shape)
    #out = output.detach().cpu().numpy()
    #    
    #pl = pv.Plotter(point_smoothing = True, shape=(2, 6))
    #fs = 12
    #i_group = 0
    #ipdb = 0
    #
    #for i in range(2):
    #    for j in range(6):
    #        
    #        if i_group < 11:
    #            ch = out[0,i_group,:,:,:]#.cpu().numpy()
    #            text = tp_name + str(start)+": " + Agroup_names[i_group]  
    #            
    #            pl.subplot(i, j)
    #            pl.add_text(text, position = 'upper_left', font_size = fs)
    #            pl.add_volume(ch, cmap = "viridis_r", opacity = "linear")
    #            
    #        i_group = i_group + 1
    #
    #pl.add_axes()
    #pl.show()
   
