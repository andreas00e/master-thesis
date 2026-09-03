from typing import List

import torch 
import torch.nn as nn 

class UncertaintyLossWraper(nn.Module): 
    def __init__(self, num_modalities: int=2) -> None:
        super().__init__()
        
        self.num_modalities = num_modalities
        self.log_vars = nn.Parameter(torch.zeros(self.num_modalities))
        
    def  forward(self, losses: List[torch.Tensor]) -> torch.Tensor: 
        losses = torch.stack(losses, dim=0) # [num_modalities, 1]
        
        precisions = torch.exp(-self.log_vars) # [num_modalities, 1]
        weighted_losses = 1/2 * precisions * losses + 1/2 * self.log_vars # [num_modalities, 1]
        
        loss = torch.sum(weighted_losses, dim=0) # []
        
        return loss 