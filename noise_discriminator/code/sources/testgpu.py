import torch

GPU = torch.cuda.is_available()

print(GPU)