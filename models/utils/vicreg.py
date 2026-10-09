import wandb 
import numpy as np 
from typing import Dict, Tuple, Optional

import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torchtyping import TensorType

import lightning.pytorch as pl 


class VICReg(nn.Module): 
# Adapted from: https://arxiv.org/abs/2105.04906
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
    
    def _invariance_loss(self, x: TensorType["n", "d"], x_: TensorType["n", "d"]) -> torch.Tensor: 
        n = x.shape[0]
        
        out = (1 / n) * ((x - x_) ** 2).sum() # []
        
        return out
        
    def _variance_loss(self, x: TensorType["n", "d"]) -> torch.Tensor: 
        var = x.var(dim=0, unbiased=False) # [d]
        std = (var + self.eps).sqrt() # [d]
        out = F.relu(self.gamma - std).mean() # []

        return out, var
    
    def _covariance_loss(self, x: TensorType["n", "d"]) -> torch.Tensor: 
        n, d = x.shape 
        if n <= 1: 
            raise ValueError(f"Calculating the sample variance requires more than one sample, got {n} sample(s)")
        
        x_centered = x - x.mean(dim=0)
        cov = (x_centered.T @ x_centered) / (n - 1) # [d, d]: sample variance
        off_diag = ~torch.eye(d, dtype=torch.bool, device=x.device)
        out = (cov[off_diag] ** 2).sum() / d
        
        return out, cov
    
    def forward(self, z: TensorType["n", "d"], z_: TensorType["n", "d"]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]: 
        inv_loss = self._invariance_loss(z, z_) # []
        
        var_loss_z, var = self._variance_loss(z)
        var_loss_z_, var_ = self._variance_loss(z_)
        var_loss = var_loss_z + var_loss_z_ # []
        
        cov_loss_z, cov = self._covariance_loss(z)
        cov_loss_z_, _ = self._covariance_loss(z_) 
        cov_loss = cov_loss_z + cov_loss_z_ # []
        tot_loss = self.lambda_ * inv_loss + self.mu * var_loss + self.nu * cov_loss # []
        
        if all([self.logger is not None, isinstance(self.logger, pl.loggers.WandbLogger), self.training]): 
            if self._global_step % self.log_every_n_steps == 0:
                with torch.no_grad(): 
                    cov_np = cov.detach().cpu().numpy()

                    if cov_np.shape[0] > 256:
                        step_stride = cov_np.shape[0] // 256
                        cov_viz = cov_np[::step_stride, ::step_stride]
                    else:
                        cov_viz = cov_np
                    
                    cov_viz = (cov_viz - np.min(cov_viz)) /(np.max(cov_viz) - np.min(cov_viz) + 1e-12) * 255
                    cov_viz = cov_viz.astype(np.uint8)
                    
                self.logger.experiment.log({
                    "train/var_histogram_z": wandb.Histogram(var.detach().cpu().numpy()), 
                    "train/var_histogram_z_": wandb.Histogram(var_.detach().cpu().numpy()), 
                    "train/covariance_matrix_z": wandb.Image(cov_viz, caption=f"Covariance Matrix step {self._global_step}")
                })
                                
        self._global_step += 1 
                
        logs_ = {
            "inv_loss": self.lambda_ * inv_loss.detach().cpu(), 
            "var_loss": self.mu * var_loss.detach().cpu(), 
            "cov_loss":  self.nu * cov_loss.detach().cpu() 
        }
             
        return tot_loss, logs_