import os
import sys
import torch
from TorchProteinLibrary.FullAtomModel import PDB2CoordsUnordered, Coords2TypedCoords

def get_group_weights(pdb_path, pdb_ids, i_type=11):
    """
    This function invokes modules from TorchPotentialLibrary and
    reads .pdb inputs. It calculates weights per group using set of structures. 

    output: torch tensor(i_type)

    """

    path_list = [pdb_path + ids + str(1) + ".pdb" for ids in pdb_ids]
         
    pdb2coords = PDB2CoordsUnordered()
    assignTypes = Coords2TypedCoords(i_type)

    coords, _, resnames, _, atomnames, num_atoms = pdb2coords(path_list)
    _, num_atoms_of_type = assignTypes(coords.to(dtype=torch.float32), resnames, atomnames, num_atoms)
        
    Natoms = num_atoms_of_type.sum().type(torch.float32)
    atoms_per_group = num_atoms_of_type.sum(axis=0) 
    weights = (Natoms / atoms_per_group).sqrt() 
    weights = (weights / weights.sum()) 
             
    return  weights 



if __name__=='__main__':

    #pdb_path  = "/u1/home/tr443/Projects/ProteinQure/data/Trajectories/"
    pdb_path  = "/home/talant/Projects/ProteinQure/data/TrajectoriesAll/"
    #pdb_path ="/projects/ccib/lamoureux/tr443/ProQureData/Trajectories/"

    AAOneLetter = ["A", "C", "D", "E", "F", "G", "H", "I", "K", "L",
                   "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y"]
    pdb_ids = []
    for el in AAOneLetter[0:19]: #skip Y for now
        for i in AAOneLetter:
            pdb_ids.append('A' + 'X' + el + '/' +'A' + i + el + '/' + 'A' + i + el)
   
    weights = get_group_weights(pdb_path, pdb_ids, 26) 
    print("weights per group -- ", weights)
