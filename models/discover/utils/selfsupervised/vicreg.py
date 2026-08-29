import torch 
import torch.nn as nn 
import torch.nn.functional as F

from torchtyping import TensorType

class VICReg(nn.Module): 
    def __init__(
        self, 
        lambda_: float, 
        mu: float, 
        nu: float, 
        gamma: float, 
        epsilon: float
        ) -> None:
        super().__init__()
        
        
        self.lambda_ = lambda_
        self.mu = mu 
        self.nu = nu
        self.gamma = gamma
        self.epsilon = epsilon 
        
    
    def _variance_loss(self, x: TensorType["n", "d"]) -> TensorType[""]: 
         
        gamma = torch.tensor(self.gamma, dtype=x.dtype, device=x.device)

        s = torch.sqrt(torch.var(x, dim=-1) + self.epsilon) # [n]. 
        out = 1 / x.shape[-1] * torch.sum(F.relu(gamma - s)) # []
        
        return out 
    
    def _covariance_loss(self, x: TensorType["n", "d"]) -> TensorType[""]: 
        cov = torch.cov(x.T, correction=1) # [d, n]: sample variance
        off_diag = ~torch.eye(cov.shape[0], dtype=torch.bool, device=cov.device)
        out = torch.sum(cov[off_diag])
        
        return out 
    
    def forward(self, z: TensorType["n", "d"], z_: TensorType["n", "d"]) -> TensorType[""]: 
        inv_loss = F.mse_loss(z, z_) # []
        var_loss = self._variance_loss(z) + self._variance_loss(z_) # []
        cov_loss = self._covariance_loss(z) + self._covariance_loss(z_) # []
        
        return self.lambda_ * inv_loss + self.mu * var_loss + self.nu * cov_loss # []