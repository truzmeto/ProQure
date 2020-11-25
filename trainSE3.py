import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

from src.Util.volume import get_volume
from src.SE3Model.EncDecSE3A import Encode, Decode
from src.Util.util import SampleBatchMix
#from src.Loss.loss_fns import XL2Loss, XL1Loss


def get_inp(pdb_ids, pdb_path, dim, rotate = True):
    """
    This function takes care of all inputs: 
    input volume <--> output volume, + rotations etc..

    """
    
    norm = False
    resolution = 1.000
    box_size = int(dim*resolution)
    batch_list = [pdb_path + ids[:3] + "/" + ids + ".pdb" for ids in pdb_ids]

    
    with torch.no_grad():

        inp, _, ori = get_volume(path_list = batch_list, 
                                 box_size = box_size,
                                 resolution = resolution,
                                 norm = norm,
                                 rot = False,
                                 trans = False)
        
        
        if rotate:
            irot = torch.randint(0, 24, (1,)).item()
            inp = Rot90Seq(inp, iRot = irot)
                    
    return inp, ori 


def run_model(volume, target, model1, model2, criterion, train = True):
    
    if train:
        model1.train();       model2.train()    
        model1.zero_grad();   model2.zero_grad()
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
        
    lrt = 0.0001
    max_epoch = 30000
    start = 0
    dev_id = 0    
    n_tripeps = 4
    n_samples = 10
    batch_size = n_tripeps * n_samples
    dim = 24
    pdb_ids = ["AAA", "ACA", "ADA", "AEA", "AFA", "AGA", "AHA", "AIA", "AKA", "ALA",
               "AMA", "ANA", "APA", "AQA", "ARA", "ASA", "ATA", "AVA", "AWA", "AYA"]
    
    trainI = list(range(1,201))
    validI = list(range(201,401))
  
    out_path = 'output/'
    params_file_name = 'net_paramsSE'
    pdb_path  = "/u1/home/tr443/Projects/ProteinQure/data/Trajectories/"

    inp_channels = 11
    out_channels = 11
    lmax = 0
    k_size = 3
    m = 8 #multiplier

    
    torch.cuda.set_device(dev_id)
    modelEncode = Encode(size=k_size, mult=m, lmax=lmax, inp_channels=inp_channels).cuda()
    modelDecode = Decode(size=k_size, mult=m, lmax=lmax, out_channels=out_channels).cuda()
    criterion =  nn.L1Loss() #XL2Loss()

    #uncomment line below if need to load saved parameters
    #model.load_state_dict(torch.load(out_path +str(start)+params_file_name))#+".pth"))

    optimizer = optim.Adam([{'params': modelEncode.parameters()},
                            {'params': modelDecode.parameters()} ], lr = lrt)#, weight_decay = wd)
    
   
    iStart = start + 1
    iEnd = max_epoch + 1
    
    for epoch in range(iStart, iEnd):

        train_list = SampleBatchMix(n_samples, n_tripeps, pdb_ids, sample_ids = trainI, shuffle = True)
        valid_list = SampleBatchMix(n_samples, n_tripeps, pdb_ids, sample_ids = validI, shuffle = True)
        
        volume, _ = get_inp(train_list, pdb_path, dim, rotate = False)
        lossT = run_model(volume, volume, modelEncode, modelDecode, criterion, train = True)    

        volumeV, _ = get_inp(valid_list, pdb_path, dim,rotate = False)
        lossV = run_model(volumeV, volumeV, modelEncode, modelDecode, criterion, train = False)  
       
        if epoch % 1000 == 0:
            torch.save({'modelEncode': modelEncode.state_dict(), 'modelDecode': modelDecode.state_dict(),},
                       out_path + str(epoch) + params_file_name)
         
        if epoch % 10 == 0:
            with open(out_path + 'log_train_valSE3.txt', 'a') as fout:
                fout.write('%d\t%f\t%f\n'%(epoch, lossT, lossV))

       
