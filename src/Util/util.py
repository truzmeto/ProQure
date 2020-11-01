import numpy as np
import random
import sys




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


