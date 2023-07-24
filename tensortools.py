import torch
import numpy as np

def normalize(x):
    return x/x.norm()

def normalize_max(x):
    return x/x.max()
    
def interp(x, xp, fp):
    fp = fp.view(-1, fp.shape[-1])
    return torch.stack([
        torch.tensor(np.interp(
        x.numpy(), xp.numpy(), row.view(-1).numpy(),
        left=0, right=0,
        )).float()
        for row in fp
    ])
