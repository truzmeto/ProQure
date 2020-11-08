import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

from src.Util.volume import get_volume
from src.Model.EncDec1 import EDNet

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


def run_model(volume, target, model, criterion, train = True):
    
    if train:
        model.train()
        model.zero_grad()
    else:
        model.eval()

        
    output = model(volume)
    L1 = criterion(output, target)
   
    L1_reg = 0.0
    for w in model.parameters():
        L1_reg = L1_reg + w.norm(1)
   
    Lambda = 0.000001
    loss = L1 + Lambda * L1_reg
   
    if train:
        loss.backward()
        optimizer.step()
    
    return L1.item()


if __name__=='__main__':

    
    
    lrt = 0.0001
    max_epoch = 10000
    start = 0
    dev_id = 1    
    batch_size = 40
    
    pdb_ids = ["AAA", "ACA", "ADA", "AEA", "AFA", "AGA", "AHA", "AIA", "AKA", "ALA", "AMA", "ANA", "APA", "AQA", "ARA"]
    tp_name = random.sample(pdb_ids, 1)[0] 
    pdb_path  = "/u1/home/tr443/Projects/ProteinQure/data/Trajectories/" + tp_name + "/" + tp_name

    trainI = list(range(200))
    validI = list(range(201,400))
  
    out_path = 'output/'
    params_file_name = 'net_params'
    dim = 16
    
    torch.cuda.set_device(dev_id)
    model = EDNet(in_dim = 11, out_dim = 11, size = 3, mult = 8).cuda()

    criterion = nn.L1Loss()

    #uncomment line below if need to load saved parameters
    #model.load_state_dict(torch.load(out_path +str(start)+params_file_name))#+".pth"))

    
    optimizer = optim.Adam(model.parameters(), lr = lrt)#, weight_decay = wd)
    iStart = start + 1
    iEnd = max_epoch + 1
    
    for epoch in range(iStart, iEnd):

        train_list = random.sample(trainI, batch_size)
        valid_list = random.sample(validI, batch_size)

        volume = get_inp(train_list, pdb_path, dim, rotate = False)
        lossT = run_model(volume, volume, model, criterion, train = True)    

        volumeV = get_inp(valid_list, pdb_path, dim,rotate = False)
        lossV = run_model(volumeV, volumeV, model, criterion, train = False)  
       
        if epoch % 200 == 0:
            torch.save(model.state_dict(), out_path + str(epoch) + params_file_name)
         
        if epoch % 5 == 0:
            with open(out_path + 'log_train_val.txt', 'a') as fout:
                fout.write('%d\t%f\t%f\n'%(epoch, lossT, lossV))

    
       
