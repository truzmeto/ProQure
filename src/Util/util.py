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


            
def SampleBatchMix(n_samples, n_tripeps, pdb_ids, sample_ids, shuffle = True):
    """
 

    """

    if shuffle:
        random.shuffle(pdb_ids)
    
    tp_list = random.sample(pdb_ids, n_tripeps)
    sample_list = random.sample(sample_ids, n_samples)

    batch_list = []
    for i in tp_list:
        for j in sample_list:
            batch_list.append(i + str(j))
            
  

    return batch_list


