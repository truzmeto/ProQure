import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

from src.Util.volume import GetVolume
from src.SE3Model.Elem2Elem import Encode, Decode
#from src.SE3Model.Elem2Charmm import Encode, Decode
from src.Util.util import SampleBatchMix
from src.Loss.loss_fns import WeMSE 
from src.Util.group_weight import get_group_weights


def run_model(volume, target, model1, model2, criterion, train = True):
    
    if train:
        model1.train();     model2.train()    
        model1.zero_grad(); model2.zero_grad()
    else:
        model1.eval()
        model2.eval()

    volume = torch.einsum('tixyz->txyzi', volume) #permute
    latent = model1(volume)

    output = model2(latent)
    output = torch.einsum('txyzi->tixyz', output) #unpermute

    loss = criterion(output, target)

    if train:
        loss.backward()
        optimizer.step()

    return loss.item()


if __name__=='__main__':
        
    dev_id = 0    
    lrt = 0.001
    max_epoch = 200
    start = 0
    n_samples = 100
    batch_size = n_samples
    dim = 20
    resolution = 1.0
    g_type = 4 # Element

    
    AAOneLetter = ["A", "C", "D", "E", "F", "G", "H", "I", "K", "L",
                   "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y"]
    pdb_ids = []
    for el in AAOneLetter[0:19]: #skip Y for now
        for i in AAOneLetter:
            pdb_ids.append('A' + 'X' + el + '/' +'A' + i + el + '/' + 'A' + i + el)

    out_path = 'output/'
    params_file_name = 'Elem2ElemParams'
    #pdb_path  = "/home/talant/Projects/ProteinQure/data/TrajectoriesAll/"
    pdb_path ="/projects/ccib/lamoureux/tr443/ProQureData/Trajectories/"             
    

    trainI = list(range(1,400))
    validI = list(range(401,800))
    inp_channels = 4
    out_channels = 4
    k_size = 5
    
    torch.cuda.set_device(dev_id)
    modelEncode = Encode(size=k_size, n_reps=[1,1], inp_channels=inp_channels, fp=False).cuda()
    modelDecode = Decode(size=k_size, n_reps=[1,1], out_channels=out_channels, fp=False).cuda()

    #-- weight initilization
    #modelEncode.apply(init_weights)
    #modelDecode.apply(init_weights)
    
    #-- uncomment line below if need to load saved parameters
    #checkpoint = torch.load(out_path + str(start) + params_file_name)
    #modelEncode.load_state_dict(checkpoint['modelEncode'])
    #modelDecode.load_state_dict(checkpoint['modelDecode'])
    
    weights = get_group_weights(pdb_path, pdb_ids, i_type=g_type).cuda()  
    criterion = WeMSE(weights) # weigted L2  
    optimizer = optim.Adam([{'params': modelEncode.parameters()},
                            {'params': modelDecode.parameters()} ], lr = lrt)
    
    GetV = GetVolume(dim, resolution, i_type=g_type)
    iStart = start + 1
    iEnd = max_epoch + 1
    spacing ="\t"
    pdbs = pdb_ids
    d_size = len(pdb_ids)
    
    ####-------- Initiate training ------------####
    for epoch in range(iStart, iEnd):

        train_list = random.sample(trainI, n_samples)
        valid_list = random.sample(validI, n_samples)

        lt = 0; lv = 0
        for i in pdb_ids:
            #--get batch
            t_list = [pdb_path + i + str(ids) + ".pdb" for ids in train_list]
            v_list = [pdb_path + i + str(ids) + ".pdb" for ids in valid_list]

            #-- project and get grids
            volume, _ = GetV.load_batch(t_list, rot=True)
            lossT = run_model(volume, volume, modelEncode, modelDecode, criterion, train = True)    
            volumeV, _ = GetV.load_batch(v_list, rot=True)
            lossV = run_model(volumeV, volumeV, modelEncode, modelDecode, criterion, train = False)  

            lt += lossT
            lv += lossV
            
        if epoch % 25 == 0:
            torch.save({'modelEncode': modelEncode.state_dict(), 'modelDecode': modelDecode.state_dict(),},
                       out_path + str(epoch) + params_file_name)
            
        with open(out_path + 'log_train_val.txt', 'a') as fout:
            fout.write('%d\t%f\t%f\n'%(epoch, lt/d_size, lv/d_size))
