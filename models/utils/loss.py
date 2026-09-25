from typing import List

import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torchtyping import TensorType

# https://arxiv.org/abs/1705.07115
class UncertaintyWeighting(nn.Module): 
    def __init__(self, num_losses: int=6) -> None:
        super().__init__()
        
        self.num_losses = num_losses
        self.log_vars = nn.Parameter(torch.zeros(self.num_losses, dtype=torch.float32))
        
    def  forward(self, losses: List[torch.Tensor]) -> torch.Tensor: 
        losses = torch.stack(losses) # [num_losses]
        
        precisions = torch.exp(-self.log_vars) # [num_losses]
        weighted_losses = 1/2 * precisions * losses + 1/2 * self.log_vars # [num_losses]
        
        loss = torch.sum(weighted_losses) # []
        
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
        self.T = T
    
    def forward(self, losses: TensorType["2", "n_losses"]) -> TensorType["n_losses"]:
        if len(losses) != self.n_losses: 
            raise ValueError(f"DWA considers last {self.n_losses} losses only, got last {len(losses)} losses.")
        
        w_k = losses[:, 0] / losses[:, 1] # [n_losses]
        w_k = torch.exp(w_k / self.T) # [n_losses]
        
        lambda_k = w_k / torch.sum(w_k) # [n_losses]

        return lambda_k 
    
    
class TimeContrastiveLoss(nn.Module): 
    def __init__(
        self, 
        temp_tcn: float
        ) -> None:   
        super().__init__() 
    
        self.temp_tcn = temp_tcn 
        self.sim = nn.CosineSimilarity()

    def forward(self, x: TensorType["batch", "3", "d_model"]) -> torch.Tensor: 
        p_ix = x[:, 0, :]
        p_iy = x[:, 1, :]
        p_iz = x[:, 2, :]
    
        tcn_nom = torch.exp(self.sim(p_ix, p_iy) / self.temp_tcn)
        tcn_denom = tcn_nom + torch.exp(self.sim(p_ix, p_iz) / self.temp_tcn)
        tcn_loss = - torch.sum(torch.log(tcn_nom / tcn_denom), dim=0)
        
        return tcn_loss
    
class TimeSmoothingLoss(nn.Module): 
    def __init__() -> None: 
        super().__init__() 
    
    def forward(self, x: TensorType["batch", "3", "d_model"]):
        x_plus = x[:, 1, :]
        x_minus = x[:, 2, :]
        x = x[:, 0, :]

        loss = F.mse_loss(x, x_minus, reduction="none") + F.mse_loss(x_plus, x, reduction="none")
        loss = torch.mean(loss)
        
        return loss        