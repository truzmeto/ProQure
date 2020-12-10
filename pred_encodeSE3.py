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
from src.SE3Model.EncDecSE3 import Encode
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
    batch_size = 1
    dim = 20
    start = 1501
    end = start + batch_size
    test_list = list(range(start, end))
    pdb_ids = ["AYA"]
    tp_name = pdb_ids[0]
    
    pdb_path  = "/u1/home/tr443/Projects/ProteinQure/data/Trajectories/" + tp_name + "/" + tp_name
    out_path = 'output/'
    params_file_name = str(118000) + 'net_paramsSE'
    inp_channels = 11
    lmax = 1
    k_size = 3
    m = 8 #multiplier

   
    torch.cuda.set_device(dev_id)
    modelEncode = Encode(size=k_size, mult=m, lmax=lmax, inp_channels=inp_channels).cuda()
    checkpoint = torch.load(out_path + params_file_name)
    modelEncode.load_state_dict(checkpoint['modelEncode'])
    
    volume, cent = get_inp(test_list, pdb_path, dim, rotate = False)
    volume = torch.einsum('tixyz->txyzi', volume) #permute
    
    modelEncode.eval()
    latent = modelEncode(volume)
    torch.save(latent, out_path + "/" + tp_name + "/" + "latentSE3" + str(start) + "-" + str(end-1) + ".pt")

    #save latent as cube file
    #=====================================================================
    #batch_id = 0
    #latent = latent.squeeze().cpu().detach().numpy()
    #print(latent.shape)
    #vec = grid2vec((dim,dim,dim), latent)
    #cube_path = out_path + "cube/" 
    #out_name = tp_name + str(start) + "SE3.gfe.map"
    #cent = cent.cpu().detach().numpy()
    #ori = cent[batch_id].round(3)
    #res = 1.000
    #write_map(vec,cube_path, out_name, ori, res, (dim,dim,dim))


    
    #plot latent volume feature
    ##=====================================================================
    fs = 12
    batch_id = 0
    print("Latent dim -- ",latent.shape)
    p = pv.Plotter(point_smoothing = True)
    #latent = torch.einsum('txyzi->tixyz', latent) #unpermute
    out = latent.squeeze()[:,:,:,batch_id].cpu().detach().numpy()
    #out = latent.squeeze() #.cpu().detach().numpy()
    #vec = out[:,:,:,1].pow(2) +  out[:,:,:,2].pow(2) +  out[:,:,:,3].pow(2)
    #vec_l = vec.sqrt().cpu().detach().numpy() #+ out[:,:,:,0].cpu().detach().numpy()

    
    print(out.min())
    text = tp_name + str(start)   
    p.add_text(text, position = 'upper_left', font_size = fs)
    p.add_volume(out, clim=[-0.1, .0], cmap = "viridis_r", opacity = "linear")
    #p.add_volume(out, cmap = "viridis_r", opacity = "linear")
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

    
       
