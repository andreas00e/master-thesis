import torch 
import torch.nn.functional as F

from torchtyping import TensorType

class VICReg(): 
    def __init__(
        self, 
        lmb: float, 
        mu: float, 
        nu: float, 
        gamma: float, 
        eps: float
        ) -> None:
        
        self.lmb = lmb
        self.mu = mu 
        self.nu = nu
        self.gamma = gamma
        self.eps = eps 
    
    def _variance_loss(self, x: TensorType["n", "d"]) -> TensorType[""]:  
        self.gamma = torch.tensor(self.gamma, dtype=x.dtype, device=x.device)

        s = torch.sqrt(torch.var(x) + self.eps) # [d]
        out = 1 / x.shape[-1] * torch.sum(torch.max(0, self.gamma - s)) # []
        
        return out 
    
    def _covariance_loss(self, x: TensorType["n", "d"]) -> TensorType[""]: 
        cov = torch.cov(x, correction=1) # sample variance
        out = torch.sum(torch.diagonal(cov) ** 2)
        
        return out 
    
    def forward(self, z: TensorType["n", "d"], z_: TensorType["n", "d"]) -> TensorType[""]: 
        inv_loss = F.mse_loss(z, z_) # []
        var_loss = self._variance_loss(z) + self._variance_loss(z_) # []
        cov_loss = torch.cov(z, correction=1) + self._covariance_loss(z_) # []
        
        return self.lmb * inv_loss + self.mu * var_loss + self.nu * cov_loss # []