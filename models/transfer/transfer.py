import os 
from typing import Optional 
from omegaconf import DictConfig

import torch
import torch.nn.functional as F 
import lightning.pytorch as pl

from models.discover.tse import TSE
from models.transfer.utils.sat import SAT
from models.transfer.utils.rce import RCE


class SkillTransfer(pl.LightningModule): 
    def __init__(
        self, 
        sat_kwargs: DictConfig, 
        rce_kwargs: DictConfig,
        optimizer_kwargs: DictConfig,
        tse_ckpt: os.PathLike, 
        sat_ckpt: Optional[os.PathLike]=None, 
        rce_ckpt: Optional[os.PathLike]=None,  
        
        ) -> None:
        
        super().__init__()
        self.save_hyperparameters()
       
        self.sat_kwargs = sat_kwargs
        self.rce_kwargs = rce_kwargs
        self.optimizer_kwargs = optimizer_kwargs

        self.tse_ckpt = tse_ckpt
        self.sat_ckpt = sat_ckpt 
        self.rce_ckpt = rce_ckpt
        
        if os.path.exists(self.tse_ckpt) and self.tse_ckpt.endswith(".ckpt"): 
            self.TSE = TSE.load_from_checkpoint(self.tse_ckpt)
               
        self.SAT = SAT(**self.sat_kwargs)
        self.RCE = RCE(**self.rce_kwargs)
        
        self.loss = F.mse_loss()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), **self.optimizer_kwargs.optimizer)
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, **self.optimizer_kwargs.lr_scheduler)
        
        return {
            "optimizer": optimizer, 
            "lr_scheduler": {
                "scheduler": scheduler, 
                "interval": "step"
            }
        }   
    
    def _shared_step(self, x): 
        x_hat = self.TSE(x)
        x_hat = self.SAT(x_hat)
        x_hat = self.RCE(x_hat)
        
        loss = self.loss(x, x_hat)
        
        self.log_dict({f"loss_{self.trainer.state.stage}": loss})
        
        return x 
    
    def forward(self, x):
        x = self._shared_step(x)
        return None
    
    def training_step(self, batch, batch_idx):
        x = self._shared_step(x)
        return None
    
    def validation_step(self, batch, batch_idx):
        x = self._shared_step(x)
        return None
    
    def test_step(self, batch, batch_idx):
        x = self._shared_step(x)
        return None
    
    def predict_step(self):
        x = self._shared_step(x)
        return None 