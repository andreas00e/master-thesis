from typing import List

import torch 
import torch.nn as nn 
import torch.nn.functional as F 

# https://arxiv.org/abs/1705.07115
class UncertaintyWeighting(nn.Module): 
    def __init__(self, num_modalities: int=2) -> None:
        super().__init__()
        
        self.num_modalities = num_modalities
        self.log_vars = nn.Parameter(torch.zeros(self.num_modalities))
        
    def  forward(self, losses: List[torch.Tensor]) -> torch.Tensor: 
        losses = torch.stack(losses) # [num_modalities, 1]
        
        precisions = torch.exp(-self.log_vars) # [num_modalities, 1]
        weighted_losses = 1/2 * precisions * losses + 1/2 * self.log_vars # [num_modalities, 1]
        
        loss = torch.sum(weighted_losses) # []
        
        return loss 
  
# https://arxiv.org/abs/2408.07985  
class SoftOptimalUncertaintyWeighting(nn.Module): 
    def __init__(self, temperature: float=1.0) -> None:
        super().__init__()
        self.temperature = temperature
    
    def forward(self, losses: List[torch.Tensor]) -> torch.Tensor: 
        losses = torch.stack(losses) # [num_modalities]
        
        precisions = torch.exp(losses.clone().detach() / self.temperature) # [num_modalities]
        weighted_losses = F.softmax(precisions, dim=0) * losses # [num_modalities]
    
        loss = torch.sum(weighted_losses)
        
        return loss