from typing import Dict, List

import torch
from torchtyping import TensorType
  
def collate_fn(batch: List[TensorType]) -> Dict[str, TensorType]:
    assert len(batch) > 0, "Batch has to contain at least one element!"
    keys = batch[0].keys()
    
    out = {}
    for key in keys: 
        out[key] = torch.stack([b[key] for b in batch], dim=0) # [batch, chunk, window, ...]
     
    return out