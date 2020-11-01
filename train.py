import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

#import functions from src
from src.Util.volume import get_volume
from src.Model.EncDec import EDNet

def get_inp(pdb_ids, pdb_path, rotate = True):
    """
    This function takes care of all inputs: 
    volume, maps + rotations etc..

    """
    
    norm = True
    resolution = 1.000
    dim = 32 #?
    box_size = int(dim*resolution)

    batch_list = [pdb_path + ids + ".pdb" for ids in pdb_ids]

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
    max_epoch = 1000
    start = 0
    dev_id = 1    

    pdb_path = "../data/" 
    pdb_ids = ["GLU-THR-LEU"]
    pdb_val = ["GLU-THR-LEU"]
    
    out_path = 'output/'
    params_file_name = 'net_params'
   
    torch.cuda.set_device(dev_id)
    model = EDNet().cuda()
    criterion = nn.L1Loss()

    #uncomment line below if need to load saved parameters
    #model.load_state_dict(torch.load(out_path +str(start)+params_file_name))#+".pth"))

    
    optimizer = optim.Adam(model.parameters(), lr = lrt)#, weight_decay = wd)
    iStart = start + 1
    iEnd = max_epoch + 1
    
    for epoch in range(iStart, iEnd):
            
        volume = get_inp(pdb_ids, pdb_path, rotate = False)
        lossT = run_model(volume, volume, model, criterion, train = True)    
                          
        volumeV = get_inp(pdb_val, pdb_path, rotate = False)
        lossV = run_model(volumeV, volumeV, model, criterion, train = False)  
       
        if epoch % 200 == 0:
            torch.save(model.state_dict(), out_path + str(epoch) + params_file_name)
         
        if epoch % 5 == 0:
            with open(out_path + 'log_train_val.txt', 'a') as fout:
                fout.write('%d\t%f\t%f\n'%(epoch, lossT, lossV))

    
       
