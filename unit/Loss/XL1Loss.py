import os
import sys
import torch
import pyvista as pv
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from src.Loss.loss_fns import XL1Loss
from src.Loss.loss_fns import XL2Loss

L1 = XL1Loss()
L2 = XL2Loss()

x = torch.rand(3,3)*5
print("X -- ", x)
y = torch.rand(3,3)*5
y[0,0]=0; y[1,1]=0; y[2,2]=0
print("Y -- ", y)

loss1 = L1(x,y)
loss2 = L2(x,y)

print("XL1 -- ", loss1)
print("XL2 -- ", loss2)

