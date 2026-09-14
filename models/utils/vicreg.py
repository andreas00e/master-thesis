import wandb 
from typing import Optional
import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torchtyping import TensorType

import lightning.pytorch as pl 


# https://arxiv.org/abs/2105.04906
class VICReg(nn.Module): 
    def __init__(
        self, 
        logger: Optional[pl.loggers.WandbLogger], 
        log_every_n_steps: int, 
        lambda_: float, 
        mu: float, 
        nu: float, 
        gamma: float,
        eps: float, 
        ) -> None:
        super().__init__()
        
        self.logger = logger
        self.log_every_n_steps = log_every_n_steps
        self._global_step = 0
        
        self.lambda_ = lambda_
        self.mu = mu 
        self.nu = nu
        self.gamma = gamma
        self.eps = eps
        
    def _variance_loss(self, x: TensorType["n", "d"]) -> torch.Tensor: 
        var = torch.var(x, dim=0) # [d]
        std = torch.sqrt(var + self.eps) # [d]
        out = torch.mean(F.relu(self.gamma - std)) # []
        
        if self.logger is not None: 
            if self._global_step % self.log_every_n_steps == 0: 
                self.logger.experiment.log({
                    "train/var_histogram": wandb.Histogram(var.detach().cpu().numpy())
                })
        
        self._global_step += 1 
        
        return out
    
    def _covariance_loss(self, x: TensorType["n", "d"]) -> torch.Tensor: 
        n, d = x.shape 
        if n <= 1: 
            raise ValueError(f"Calculating the sample variance requires more than one sample, got {n} sample(s)")
        
        x_centered = x - x.mean(dim=0)
        cov = (x_centered.T @ x_centered) / (n - 1) # [d, d]: sample variance
        off_diag = ~torch.eye(d, dtype=torch.bool, device=x.device)
        out = torch.sum(cov[off_diag] ** 2) / d
        
        return out 
    
    def forward(self, z: TensorType["n", "d"], z_: TensorType["n", "d"]) -> torch.Tensor: 
        inv_loss = F.mse_loss(z, z_) # []
        var_loss = self._variance_loss(z) + self._variance_loss(z_) # []
        cov_loss = self._covariance_loss(z) + self._covariance_loss(z_) # []
        tot_loss = self.lambda_ * inv_loss + self.mu * var_loss + self.nu * cov_loss # []
        
        logs_ = {
            "inv_loss": inv_loss.detach(), 
            "var_loss": var_loss.detach(), 
            "cov_loss": cov_loss.detach() 
        }
     
        return tot_loss, logs_