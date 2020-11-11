import os
import sys
import torch
import pyvista as pv
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from src.Loss.loss_fns import WLoss

L = WLoss()
x = torch.rand(2,2,10,10,10)
y = torch.rand(2,2,10,10,10)
loss = L(x,y)

print(loss.item())
