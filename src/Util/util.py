import numpy as np
import random
import sys


def write_map(vec, out_path, out_name, ori, res, n):
    """
    This function outputs a .map file
    that is readeable by vmd.
    """

    fname = out_path + out_name
    nx, ny, nz = n
    spacing = res      
    
    with open(fname,'w') as fp:

        fp.write("GRID_PARAMETER_FILE\n") 
        fp.write("GRID_DATA_FILE\n") 
        fp.write("MACROMOLECULE\n") 
        fp.write("SPACING %5.3f\n" % res)
        fp.write("NELEMENTS "+str(nx-1)+" "+str(ny-1)+ " "+str(nz-1)+"\n") #consider changing!!!!! 
        fp.write("CENTER "+str(ori[0])+" "+str(ori[1])+" "+str(ori[2])+"\n")

        for i in range(nx*ny*nz):
            fp.write("%10.3f\n" % vec[i])

            
def vec2grid(n, vec):
    """
    This function transforms 1D vec. into
    tensor (nx,ny,nz) data structure
    """
    nx, ny, nz = n
    grid = np.reshape(vec, newshape = (nx,ny,nz), order = "F")

    return grid


def grid2vec(n, grid):
    """
    This function transforms 3D grid.(nx,ny,nz) into
    vector (nx*ny*nz)
    """
    nx, ny, nz = n
    vec = np.reshape(grid, newshape = nx*ny*nz, order = "F")

    return vec


            
def sample_batch(batch_size, pdb_ids, pdb_path, shuffle = True):
    """
    Function to produce random sample of pdb ids from a given list.
    This ids are joined with the path to actual file location
    and used to access the data.

    Input:  batch_size  - number of input files in a batch, dtype=list
            pdb_ids     - full list of pdbs to sample from, dtype=list
            pdb_path    - path to file location, dtype=string

    Output: batch_list  - list of selected files as full path, dtype=list
            pdb_list    - list of selected pdbs, dtype=list

    """

    if batch_size > len(pdb_ids):
        raise ValueError("Batch size must be less or equal to #pdbs")

    if shuffle:
        random.shuffle(pdb_ids)

    pdb_list = random.sample(pdb_ids, batch_size)
    batch_list = [pdb_path+ids+".pdb" for ids in pdb_list]

    return batch_list, pdb_list


