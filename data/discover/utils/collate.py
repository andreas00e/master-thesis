from typing import Dict, List, Tuple

import torch
from torchtyping import TensorType
  
def collate_fn(batch: List[Tuple[TensorType["window", "..."], ...]]) -> Dict[str, TensorType["window", "batch", "..."]]:
    
    rgb_obs = torch.stack([x[0] for x in batch], dim=1) 
    rgb_obs_plus = torch.stack([x[1] for x in batch], dim=1) 
    
    return {
        "rgb_obs": rgb_obs, 
        "rgb_obs_plus": rgb_obs_plus
    }