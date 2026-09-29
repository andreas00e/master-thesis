from typing import Tuple 

import torch 
import torch.nn.functional as F 
from torchtyping import TensorType

def _pair_metrics(a: TensorType["n", "d_model"], b: TensorType["n", "d_model"]) -> Tuple[TensorType["n"], ...]:    
    print(a.shape)
    
    a, b = F.normalize(a, dim=-1), F.normalize(b, dim=-1)
    
    pos = torch.sum((a * b), dim=-1) # [n]
    neg = torch.sum((a * torch.roll(b, shifts=1, dims=0)), dim=-1) # [n]

    sim = a @ b.T # [n, n]
    top_one = (torch.argmax(sim, dim=1) == torch.arange(a.shape[0], device=a.device)).float() # [n]
    
    return pos, neg, top_one