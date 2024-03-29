import os
import sys
import torch
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from src.Loss.loss_fns import PenLoss

L = PenLoss()
x = torch.rand(2,2,10,10,10)
y = torch.rand(2,2,10,10,10)
loss = L(x,y, thresh=1.0)

print(loss.item())
