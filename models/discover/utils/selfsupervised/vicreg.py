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

        s = torch.sqrt(torch.var(x, dim=0, correction=1) + self.epsilon) # [d]. 
        out = torch.mean(F.relu(gamma - s)) # []
        
        return out 
    
    def _covariance_loss(self, x: TensorType["n", "d"]) -> TensorType[""]: 
        n, d = x.shape 
        if n <= 1: 
            raise ValueError(f"Calculating the sample variance requires more than one sample, got {n} sample(s)")
        
        x_centered = x - x.mean(dim=0)
        cov = (x_centered.T @ x_centered) / (n - 1) # [d, d]: sample variance
        off_diag = ~torch.eye(cov.shape[0], dtype=torch.bool, device=x.device)
        out = torch.sum(cov[off_diag] ** 2) / d
        
        return out 
    
    def forward(self, z: TensorType["n", "d"], z_: TensorType["n", "d"]) -> TensorType[""]: 
        inv_loss = F.mse_loss(z, z_) # []
        var_loss = self._variance_loss(z) + self._variance_loss(z_) # []
        cov_loss = self._covariance_loss(z) + self._covariance_loss(z_) # []
        
        return self.lambda_ * inv_loss + self.mu * var_loss + self.nu * cov_loss # []