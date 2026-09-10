from typing import List

import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torchtyping import TensorType

# https://arxiv.org/abs/1705.07115
class UncertaintyWeighting(nn.Module): 
    def __init__(self, num_losses: int=2) -> None:
        super().__init__()
        
        self.num_losses = num_losses
        self.log_vars = nn.Parameter(torch.zeros(self.num_losses, dtype=torch.flaot32))
        
    def  forward(self, losses: List[torch.Tensor]) -> torch.Tensor: 
        losses = torch.stack(losses, dim=0) # [num_modalities, 1]
        
        precisions = torch.exp(-self.log_vars) # [num_modalities, 1]
        weighted_losses = 1/2 * precisions * losses + 1/2 * self.log_vars # [num_modalities, 1]
        
        loss = torch.sum(weighted_losses, dim=0) # []
        
        return loss 
  
# https://arxiv.org/abs/2408.07985  
class SoftOptimalUncertaintyWeighting(nn.Module): 
    def __init__(self, temperature: float=1.0) -> None:
        super().__init__()
        self.temperature = temperature
    
    def forward(self, losses: List[torch.Tensor]) -> torch.Tensor: 
        losses = torch.stack(losses, dim=0) # [num_modalities]
        
        precisions = torch.exp(losses.clone().detach() / self.temperature) # [num_modalities]
        weighted_losses = F.softmax(precisions, dim=0) * losses # [num_modalities]
    
        loss = torch.sum(weighted_losses, dim=0)
        
        return loss
    
# https://arxiv.org/pdf/1803.10704
class DynamicWeightAverage(nn.Module): 
    def __init__(
        self, 
        n_losses: int, 
        T: int, # softmax temperature
        ) -> None:
        super().__init__()
        
        self.n_losses = n_losses
        self.temperature = T
    
    def forward(self, losses:TensorType["2", "n_losses"]) -> TensorType["n_losses"]:
        if len(losses) != self.n_losses: 
            raise ValueError(f"DWA considers last {self.n_losses} losses only, got last {len(losses)} losses.")
        
            
        w_k = losses[:, 0] / losses[:, 1] # [n_losses]
        w_k = torch.exp(w_k / self.T) # [n_losses]
        
        lambda_k = w_k / torch.sum(w_k) # [n_losses]

        return lambda_k 