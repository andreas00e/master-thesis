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
    
def collate_discover_test(batch: List[TensorType]) -> Dict[str, TensorType["*"]]:
    if len(batch) <= 0: raise ValueError(f"Batch has to contain at least one element, got {len(batch)}.")
        
    return {
        key: pad_sequence(
            [b[key] for b in batch], 
            batch_first=True, 
            padding_value=float("nan") if batch[0][key].is_floating_point() else 0
            )
        for key in batch[0].keys()  
    }

def collate_transfer(batch: List[Dict[str, TensorType["steps", "*"]]]) -> Dict[str, TensorType["*"]]: 
    if len(batch) <= 0: raise ValueError(f"Batch has to contain at least one element, got {len(batch)}.")
    
    return {
        key: pad_sequence(
            [b[key] for b in batch], 
            batch_first=True, 
            padding_value=float("nan") if batch[0][key].is_floating_point() else 0
            )
        for key in batch[0].keys()  
    }