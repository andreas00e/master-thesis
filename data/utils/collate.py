from typing import Dict, List

import torch 
from torch.nn.utils.rnn import pad_sequence
from torchtyping import TensorType

  
def collate_discover(batch: List[TensorType]) -> Dict[str, TensorType["*"]]:
    if len(batch) <= 0: raise ValueError(f"Batch has to contain at least one element, got {len(batch)}.")
        
    return {
        key:torch.stack([b[key] for b in batch], dim=0)
        for key in batch[0].keys()
        }

def collate_transfer(batch: List[Dict[str, TensorType["steps", "*"]]]) -> Dict[str, TensorType["*"]]: 
    if len(batch) <= 0: raise ValueError(f"Batch has to contain at least one element, got {len(batch)}.")
    
    out = {}
   
    for key in batch[0].keys(): # actions, rgb_obs, joint_dsc, joint_obs 
        element: List[TensorType["steps", "*"]] = [b[key] for b in batch] # List[]        
        out[key] = pad_sequence(element, batch_first=True, padding_value=float("nan"))

    return out 