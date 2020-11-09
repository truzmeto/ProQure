import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import matplotlib.pyplot as plt

from src.Util.volume import get_volume
from src.Model.EncDec2 import Encode

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
    modelEncode = Encode(in_dim = 11, size = 3, mult = 8).cuda()
    checkpoint = torch.load(out_path + params_file_name)
    modelEncode.load_state_dict(checkpoint['modelEncode'])
    
    volume = get_inp(test_list, pdb_path, dim, rotate = False)
    modelEncode.eval()
    latent = modelEncode(volume)

    torch.save(latent, out_path + "/" + tp_name + "/" + "latent" + str(start) + "-" + str(end) + ".pt")
    
    #plot latent vector distribution
    batch_id=0
    x = latent.squeeze()[batch_id].cpu().detach().numpy() 
    plt.hist(x, bins = 30)
    plt.show()
    
    
       
