from typing import Dict, List

import torch 
from torch.nn.utils.rnn import pad_sequence
from torchtyping import TensorType

  
def collate_discover(batch: List[TensorType]) -> Dict[str, TensorType]:
    assert len(batch) > 0, "Batch has to contain at least one element!"
        
    out = {}
    for key in  batch[0].keys(): 
        out[key] = torch.stack([b[key] for b in batch], dim=0) # [batch, chunk, window, ...]
     
    return out

def collate_transfer(batch: List[Dict[str, TensorType["steps", "*"]]]) -> Dict[str, TensorType["batch", "~steps", "*"]]: 
    assert len(batch) > 0, "Batch has to contain at least one element!"
    out = {}

    for key in batch[0].keys(): # actions, rgb_obs, joint_dsc, joint_obs 
        element: List[TensorType["steps", "*"]] = [b[key] for b in batch] # List[]        
        out[key] = pad_sequence(element, batch_first=True)

    return out 