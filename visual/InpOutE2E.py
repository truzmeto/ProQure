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
from src.Util.volume import GetVolume
from src.SE3Model.Elem2Elem import Encode, Decode



if __name__=='__main__':


    dev_id = 0    
    batch_size = 1
    dim = 20
    start = 100
    pdb_ids = ["ACA"]
    tp_name = pdb_ids[0]

    pdb_path  = "/home/talant/Projects/ProteinQure/data/Trajectories/" + tp_name + "/"
    #out_path = '/u1/home/tr443/Projects/ProteinQure/ProQure/output/'
    out_path = '../output/' #'/home/talant/Projects/ProteinQure/ModelsF/exp1/output/'
    params_file_name = str(200) + 'Elem2ElemParams'
    
    inp_channels = 4
    k_size = 5
    m = 4 #multiplier
    resolution = 1.0
    i_type = 4
    path_list = [pdb_path + tp_name + str(start) + ".pdb"]

    torch.cuda.set_device(dev_id)
    modelDecode = Decode(size=k_size, n_reps=[1,1], out_channels=inp_channels).cuda()
    checkpoint = torch.load(out_path + params_file_name)
    modelDecode.load_state_dict(checkpoint['modelDecode'])
    modelEncode = Encode(size=k_size, n_reps=[1,1], inp_channels=inp_channels).cuda()
    modelEncode.load_state_dict(checkpoint['modelEncode'])

    GetV = GetVolume(dim, resolution, i_type)
    volume, _ = GetV.load_batch(path_list, rot=False)
    volume = torch.einsum('tixyz->txyzi', volume) #permute
    modelEncode.eval()
    latent = modelEncode(volume)
    modelDecode.eval()
    output = modelDecode(latent)
    output = torch.einsum('txyzi->tixyz', output) #unpermute
    volume = torch.einsum('txyzi->tixyz', volume) #unpermute


    if(i_type == 4):
        Agroup_names = ["C","N" ,"O", "S"]
    elif(i_type == 26):
        Agroup_names = 26*["A"]
    else:
        Agroup_names = ["Sulfur/Selenium"  , "Nitrogen Amide",
                        "Nitrogen Aromatic", "Nitrogen Guanidinium",
                        "Nitrogen Ammonium", "Oxygen Carbonyl",
                        "Oxygen Hydroxyl"  , "Oxygen Carboxyl",
                        "Carbon sp2"       , "Carbon Aromatic",
                        "Carbon sp3"]

    pl = pv.Plotter(point_smoothing = True, shape=(i_type, 2))
    fs = 9
    i_group = 0
    ipdb = 0


    vol = volume.detach().cpu().numpy()
    out = output.detach().cpu().numpy()

    norm = vol.max() - vol.min()
    
    for i in range(i_type):
        
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

        ch2[ch2 < 0.003]=0.0
        pl.add_volume(ch2, cmap = "viridis_r", opacity = "linear")
        print("max", ch2.max(), "   min", ch2.min())
        
pl.show(screenshot=tp_name + '.png' ,window_size=[528, 950])

    
