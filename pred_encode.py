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
from src.Model.EncDec3 import Encode
from src.Util.util import write_map, grid2vec

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
        
    dev_id = 0    
    batch_size = 5
    dim = 20
    start = 1501
    end = start + batch_size
    test_list = list(range(start, end))
    pdb_ids = ["AYA"]
    tp_name = pdb_ids[0]
    
    pdb_path  = "/u1/home/tr443/Projects/ProteinQure/data/Trajectories/" + tp_name + "/" + tp_name
    out_path = 'output/'
    params_file_name = str(50000) + 'net_params'

    torch.cuda.set_device(dev_id)
    modelEncode = Encode(in_dim = 11, size = 3, mult = 8).cuda()
    checkpoint = torch.load(out_path + params_file_name)
    modelEncode.load_state_dict(checkpoint['modelEncode'])
    
    volume, cent = get_inp(test_list, pdb_path, dim, rotate = False)
    cent = cent.cpu().detach().numpy()
    modelEncode.eval()
    latent = modelEncode(volume)
    print(latent.shape)
    torch.save(latent, out_path + "/" + tp_name + "/" + "latent" + str(start) + "-" + str(end-1) + ".pt")

    #save latent as cube file
    #=====================================================================
    batch_id = 0
    latent = latent.squeeze()[batch_id].cpu().detach().numpy()
    print(latent.shape)
    vec = grid2vec((dim,dim,dim), latent)
    cube_path = out_path + "cube/" 
    out_name = tp_name + str(start) + ".gfe.map"
    ori = cent[batch_id].round(3)
    res = 1.000
    write_map(vec,cube_path, out_name, ori, res, (dim,dim,dim))
    
        
    #plot latent volume feature
    ##=====================================================================
    fs = 12
    print("Latent dim -- ",latent.shape)
    p = pv.Plotter(point_smoothing = True)
    out = latent#.squeeze()[batch_id].cpu().detach().numpy() 
    text = tp_name + str(start)   
    p.add_text(text, position = 'upper_left', font_size = fs)
    p.add_volume(abs(out), cmap = "viridis_r", opacity = "linear")
    p.add_axes()
    p.show()

    
    #plot latent vector distribution
    ##=====================================================================
    #plot_fn = 'visual/latent_dist.pdf'
    #fig = plt.figure(figsize=(5.,3.5))
    #batch_id = 0
    #x = latent.squeeze()[batch_id].cpu().detach().numpy() 
    #plt.hist(x, bins = 30)
    #plt.title(tp_name + str(start)+" latent feature")
    #plt.xlabel("Latent vector")
    #plt.ylabel("Frequency")

    #fig.set_tight_layout(True)
    #plt.show()
    #fig.savefig(plot_fn)
    #os.system("epscrop %s %s" % (plot_fn, plot_fn))

    
       
