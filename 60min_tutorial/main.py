import torch
from .net import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

#net.to(device)