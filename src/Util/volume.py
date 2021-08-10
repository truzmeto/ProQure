import os
import sys
import torch
from TorchProteinLibrary.Volume import TypedCoords2Volume, VolumeRotation
from TorchProteinLibrary.FullAtomModel import PDB2CoordsUnordered, Coords2TypedCoords
from TorchProteinLibrary.FullAtomModel import getRandomRotation, getRandomTranslation
from TorchProteinLibrary.FullAtomModel import CoordsRotate, CoordsTranslate, getBBox


class GetVolume:
    def __init__(self, box_size, resolution, i_type, device='cuda'):

        self.pdb2coords = PDB2CoordsUnordered()
        self.assignTypes = Coords2TypedCoords(i_type)
        self.translate = CoordsTranslate()
        self.rotate = CoordsRotate()
        #self.getBBox = getBBox()
        self.project = TypedCoords2Volume(box_size, resolution)
        self.box_size = box_size
        self.resolution = resolution
        self.device = device
        self.rotate = CoordsRotate()

        
    def load_batch(self, path_list, rot=False):
        
        batch_size = len(path_list)
        coords, _, resnames, _, atomnames, num_atoms = self.pdb2coords(path_list)
        a,b = getBBox(coords, num_atoms)
        protein_center = (a+b)*0.5
        coords = self.translate(coords, -protein_center, num_atoms)
        random_rotations = getRandomRotation(batch_size)

        #rotate xyz 
        if rot:
            coords = self.rotate(coords, random_rotations, num_atoms)

        
        box_center = torch.zeros(batch_size, 3, dtype=torch.double, device='cpu').fill_(self.resolution*self.box_size/2.0)
        coords = self.translate(coords, box_center, num_atoms)

        coords, num_atoms_of_type = self.assignTypes(coords.to(dtype=torch.float32), resnames, atomnames, num_atoms)
        volume = self.project(coords.to(device=self.device), num_atoms_of_type.to(device=self.device))
    
        return volume, protein_center



def grid_rot(volume, batch_size, rot_matrix):
    """
    This function applies rotation on a grid
    given a rotation matrix. Matrix comes from
    the rotation performed on xyz for consistency
    """
    
    #with torch.no_grad():
    volume_rotate = VolumeRotation(mode = 'bilinear')
    R = rot_matrix #rotgetRandomRotation(batch_size)
    volume = volume_rotate(volume, R.to(dtype = torch.float, device = 'cuda'))

    return volume


if __name__=='__main__':

    #from volume import GetVolume
    pdb_ids = ["1ycr"]
    path = "/u1/home/tr443/data/fragData/"
    box_size = 60  
    resolution = 1.0
    path_list = [path + pdb_ids[0]]
    i_type=4
    print(path_list)

    GetV = GetVolume(box_size, resolution,i_type)
    volume, _ = GetV.load_batch(path_list)
    print(volume.shape)
