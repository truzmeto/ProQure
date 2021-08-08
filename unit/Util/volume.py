import os
import sys
import torch
import pyvista as pv
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from src.Util.volume import get_volume

pdb_ids = ["AYA1"]#, "ARA2"]
tp_name = pdb_ids[0][:3]
path  = "/u1/home/tr443/Projects/ProteinQure/data/Trajectories/" + tp_name + "/"
#path = "/u1/home/tr443/data/fragData/"

box_size = 24  
resolution = 1.0

for i in range(len(pdb_ids)):

    path_list = [path + pdb_ids[i] + ".pdb"]
    print(pdb_ids[i])

    volume, _, _ = get_volume(path_list,
                              box_size,
                              resolution,
                              norm = False,
                              rot = False,
                              trans = False)

print('Grid dim -- ', volume.shape)
print(volume.max())


Agroup_names = ["Sulfur/Selenium"  , "Nitrogen Amide",
                "Nitrogen Aromatic", "Nitrogen Guanidinium",
                "Nitrogen Ammonium", "Oxygen Carbonyl",
                "Oxygen Hydroxyl"  , "Oxygen Carboxyl",
                "Carbon sp2"       , "Carbon Aromatic",
                "Carbon sp3"]



pl = pv.Plotter(point_smoothing = True, shape=(11, 1))
fs = 9
i_group = 0
ipdb = 0

for i in range(11):
    for j in range(1):

        if i_group < 11:
            ch = volume[0,i_group,:,:,:].cpu().numpy()
            #ext = pdb_ids[ipdb] + ": " + Agroup_names[i_group]  
            text = Agroup_names[i_group]  
            
            pl.subplot(i, j)
            pl.add_text(text, position = 'upper_left', font_size = fs)
            pl.add_volume(ch, cmap = "viridis_r", opacity = "linear")

        i_group = i_group + 1

#pl.add_axes()
pl.show()

