import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import matplotlib.pyplot as plt

from src.Util.volume import get_volume
from src.SE3Model.ED import VNet
from src.Util.util import SampleBatchMix

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


def run_model(volume, target, model, criterion, train = True):
    
    if train:
        model.train()    
        model.zero_grad()
    else:
        model.eval()
    
    volume = torch.einsum('tixyz->txyzi', volume) #permute
    output = model(volume)
    output = torch.einsum('txyzi->tixyz', output) #unpermute

    loss = criterion(output, target)

    if train:
        loss.backward()
        optimizer.step()

    return loss.item()


def plot_grad_flow(named_parameters):
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().sum())
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation=70)#"vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    #plt.show()

def init_weights(m):
    for p in m.parameters():
        torch.nn.init.uniform_(p, -1., 1.)
        
    
    
if __name__=='__main__':
        
    #lrt = 0.0001
    lrt = 0.001
    
    max_epoch = 40000
    start = 10000
    dev_id = 0    
    n_tripeps = 2
    n_samples = 4
    batch_size = n_tripeps * n_samples
    dim = 32
    pdb_ids = ["AAA", "ACA", "ADA", "AEA", "AFA", "AGA", "AHA", "AIA", "AKA", "ALA"]
               #"AMA", "ANA", "APA", "AQA", "ARA", "ASA", "ATA", "AVA", "AWA", "AYA"]
    
    trainI = list(range(1,1000))
    validI = list(range(1001,1501))

  
    out_path = 'output/'
    params_file_name = 'net_paramsSE'
    pdb_path  = "/u1/home/tr443/Projects/ProteinQure/data/Trajectories/"

    inp_channels = 11
    out_channels = 11
    k_size = 3
       
    torch.cuda.set_device(dev_id)
    model = VNet(size = k_size, n_reps = [3,3]).cuda()
    #model.apply(init_weights)
    criterion =  nn.SmoothL1Loss() 

    ##uncomment line below if need to load saved parameters
    checkpoint = torch.load(out_path + str(start) + params_file_name)
    model.load_state_dict(checkpoint)

    
    optimizer = optim.Adam(model.parameters(), lr = lrt)#, weight_decay = wd)
    iStart = start + 1
    iEnd = max_epoch + 1
    fig = plt.figure(figsize=(10.,6.))
    
    for epoch in range(iStart, iEnd):

        train_list = SampleBatchMix(n_samples, n_tripeps, pdb_ids, sample_ids = trainI, shuffle = True)
        valid_list = SampleBatchMix(n_samples, n_tripeps, pdb_ids, sample_ids = validI, shuffle = True)
        
        volume, _ = get_inp(train_list, pdb_path, dim, rotate = False)
        lossT = run_model(volume, volume, model, criterion, train = True)    

        
        volumeV, _ = get_inp(valid_list, pdb_path, dim,rotate = False)
        lossV = run_model(volumeV, volume, model, criterion, train = False)  
       
        if epoch % 100 == 0:
            torch.save(model.state_dict(), out_path + str(epoch) + params_file_name)
            plot_grad_flow(model.named_parameters())
            fig.set_tight_layout(True)
            plt.savefig(out_path + str(epoch) + "grad.pdf")
            fig.clf()
            
        if epoch % 10 == 0:
            with open(out_path + 'log_train_valSE3.txt', 'a') as fout:
                fout.write('%d\t%f\t%f\n'%(epoch, lossT, lossV))

       
