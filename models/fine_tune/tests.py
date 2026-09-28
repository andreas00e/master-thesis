from typing import Tuple 

import torch 
import torch.nn.functional as F 
from torchtyping import TensorType

def _pair_metrics(a: TensorType["n", "d_model"], b: TensorType["n", "d_model"]) -> Tuple[TensorType["n"], TensorType["n"], TensorType["n", "n"]]:
    a, b = F.normalize(a, dim=-1), F.normalize(b, dim=-1) # [n, d_model], [n, d_model]
    
    pos = F.cosine_similarity(a, b, dim=-1) # [n]
    neg = F.cosine_similarity(a, torch.roll(b, shifts=3, dims=0), dim=-1) # [n]
    
    sim = a @ b.T # [n, n]
    top_one = (torch.argmax(sim, dim=1) == torch.arange(a.shape[0], device=a.device)).float()
    
    return pos, neg, top_one