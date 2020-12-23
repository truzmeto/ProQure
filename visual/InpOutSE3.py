import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import matplotlib.pyplot as plt
import pyvista as pv

sys.path.append(os.path.join(os.path.dirname(__file__), "./.."))
from src.Util.volume import get_volume
from src.SE3Model.ED import VNet


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
        inp, _, ori = get_volume(path_list = batch_list, 
                                 box_size = box_size,
                                 resolution = resolution,
                                 norm = norm,
                                 rot = False,
                                 trans = False)
    return inp, ori 


if __name__=='__main__':


    dev_id = 1    
    batch_size = 1
    dim = 32
    start = 1401
    pdb_ids = ["AYA"]
    tp_name = pdb_ids[0]

    pdb_path  = "/u1/home/tr443/Projects/ProteinQure/data/Trajectories/" + tp_name + "/" + tp_name
    out_path = '/u1/home/tr443/Projects/ProteinQure/experiments/exp20/output/'
    params_file_name = str(1000) + 'net_paramsSE'

    inp_channels = 11
    k_size = 3

    torch.cuda.set_device(dev_id)
    checkpoint = torch.load(out_path + params_file_name)
    model = VNet(size=k_size, n_reps = [3,3]).cuda()
    model.load_state_dict(checkpoint)


    volume, _ = get_inp([start], pdb_path, dim, rotate = False)
    volume = torch.einsum('tixyz->txyzi', volume) #permute
    model.eval()
    output = model(volume)
    output = torch.einsum('txyzi->tixyz', output) #unpermute
    volume = torch.einsum('txyzi->tixyz', volume) #unpermute
    
    Agroup_names = ["Sulfur/Selenium"  , "Nitrogen Amide",
                    "Nitrogen Aromatic", "Nitrogen Guanidinium",
                    "Nitrogen Ammonium", "Oxygen Carbonyl",
                    "Oxygen Hydroxyl"  , "Oxygen Carboxyl",
                    "Carbon sp2"       , "Carbon Aromatic",
                    "Carbon sp3"]


    pl = pv.Plotter(point_smoothing = True, shape=(11, 2))
    fs = 9
    i_group = 0
    ipdb = 0


    vol = volume.detach().cpu().numpy()
    out = output.detach().cpu().numpy()

    norm = vol.max() - vol.min()
    
    for i in range(11):
        
        ch1 = vol[0,i,:,:,:]
        ch2 = out[0,i,:,:,:]

        diff = ch1 - ch2
        err = diff*diff
        err = np.sqrt(err.mean()) #/norm
        err = round(err,3)
        
        text1 = Agroup_names[i]  
        text2 = "RMSE = " + str(err)  

        pl.subplot(i, 0)
        pl.add_text(text1, position = 'upper_right', font_size = fs)
        if i == 0:
            pl.add_text("Input", position = 'upper_left', font_size = fs)
        pl.add_volume(ch1, cmap = "viridis_r", opacity = "linear")

        pl.subplot(i, 1)
        pl.add_text(text2, position = 'upper_right', font_size = fs)
        if i == 0:
            pl.add_text("Prediction", position = 'upper_left', font_size = fs)
            pl.add_text(tp_name + str(start), position = 'lower_left', font_size = fs)

        ch2[ch2 < 0.07]=0.0
        pl.add_volume(ch2, cmap = "viridis_r", opacity = "linear")
        print("max", ch2.max(), "   min", ch2.min())
        
pl.show()

    
