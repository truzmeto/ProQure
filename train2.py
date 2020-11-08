import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

from src.Util.volume import get_volume
from src.Model.EncDec2 import Encode, Decode

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
        
        
        if rotate:
            irot = torch.randint(0, 24, (1,)).item()
            inp = Rot90Seq(inp, iRot = irot)
                    
    return inp 


def run_model(volume, target, model1, model2, criterion, train = True):
    
    if train:
        model1.train();       model2.train()    
        model1.zero_grad();   model2.zero_grad()
    else:
        model1.eval()
        model2.eval()

        
    latent = model1(volume)
    output = model2(latent)
        
    L1 = criterion(output, target)
   
    #L1_reg = 0.0
    #for w in model.parameters():
    #    L1_reg = L1_reg + w.norm(1)
    #Lambda = 0.000001
    #loss = L1 + Lambda * L1_reg
    loss = L1
    
    if train:
        loss.backward()
        optimizer1.step()
        optimizer2.step()
        
    return L1.item()


if __name__=='__main__':

        
    lrt = 0.0001
    max_epoch = 20000
    start = 0
    dev_id = 1    
    batch_size = 40
    
    pdb_ids = ["AAA", "ACA", "ADA", "AEA", "AFA", "AGA", "AHA", "AIA", "AKA", "ALA", "AMA", "ANA", "APA", "AQA", "ARA"]
    tp_name = random.sample(pdb_ids, 1)[0] 
    pdb_path  = "/u1/home/tr443/Projects/ProteinQure/data/Trajectories/" + tp_name + "/" + tp_name

    trainI = list(range(400))
    validI = list(range(401,600))
  
    out_path = 'output/'
    params_file_name = 'net_params'
    dim = 16
    
    torch.cuda.set_device(dev_id)
    modelEncode = Encode(in_dim = 11, size = 3, mult = 8).cuda()
    modelDecode = Decode(out_dim = 11, size = 3, mult = 8).cuda()
        
    criterion = nn.L1Loss()

    #uncomment line below if need to load saved parameters
    #model.load_state_dict(torch.load(out_path +str(start)+params_file_name))#+".pth"))

    
    optimizer1 = optim.Adam(modelEncode.parameters(), lr = lrt)#, weight_decay = wd)
    optimizer2 = optim.Adam(modelDecode.parameters(), lr = lrt)#, weight_decay = wd)
    
    iStart = start + 1
    iEnd = max_epoch + 1
    
    for epoch in range(iStart, iEnd):

        train_list = random.sample(trainI, batch_size)
        valid_list = random.sample(validI, batch_size)

        volume = get_inp(train_list, pdb_path, dim, rotate = False)
        lossT = run_model(volume, volume, modelEncode, modelDecode, criterion, train = True)    

        volumeV = get_inp(valid_list, pdb_path, dim,rotate = False)
        lossV = run_model(volumeV, volumeV, modelEncode, modelDecode, criterion, train = False)  
       
        if epoch % 500 == 0:
            torch.save({'modelEncode': modelEncode.state_dict(), 'modelDecode': modelDecode.state_dict(),},
                       out_path + str(epoch) + params_file_name)
         
        if epoch % 10 == 0:
            with open(out_path + 'log_train_val.txt', 'a') as fout:
                fout.write('%d\t%f\t%f\n'%(epoch, lossT, lossV))

    
       
