import torch 
import torch.nn.functional as F

from torchtyping import TensorType

class VicReg(): 
    def __init__(
        self, 
        lambd_: float, 
        mu: float, 
        nu: float, 
        gamma: float, 
        epsilon: float
        ) -> None:
        
        self.lambd_ = lambd_
        self.mu = mu 
        self.nu = nu
        self.gamma = gamma
        self.epsilon = epsilon 
    
    def _variance_loss(self, x: TensorType["n", "d"]): 
        _, d = x.shape
        
        s = torch.sqrt(torch.var(x, dim=-1) + self.epsilon) # [d]
        out = 1 / d * torch.sum(torch.max(0, self.gamma - s))
        return out 
    
    def _covariance_loss(self, x: TensorType["n", "d"]) -> TensorType[""]: 
        cov = torch.cov(x, correction=1) # sample variance
        out = torch.sum(torch.diagonal(cov) ** 2)
        
        return out 
    
    def forward(self, z, z_): 
        inv_loss = self.F.mse_loss(z, z_)
        var_loss = self._variance_loss(z) + self._variance_loss(z_)
        cov_loss = torch.cov(z, correction=1) + self._covariance_loss(z_)
        
        return self.lambd_ * inv_loss + self.mu * var_loss + self.nu * cov_loss