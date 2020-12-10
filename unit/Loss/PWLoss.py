import os
import sys
import torch
import pyvista as pv
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from src.Loss.loss_fns import PWLoss

L = PWLoss()
L1 = torch.nn.L1Loss()
y = torch.rand(2,2,10,10,10)
x = torch.ones(2,2,10,10,10)
loss = L(x,y)
loss1 = L1(x,y)

print("Loss",loss.item())
print('L1', loss1.item())
