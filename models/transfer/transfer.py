from typing import Dict

import torch
import lightning.pytorch as pl

from .utils.rce import RCE


class SkillTransfer(pl.LightningModule): 
    def __init__(
        self, 
        rce_kwargs: Dict,
        optimizer_kwargs: Dict 
        ) -> None:
        
        super().__init__()
        self.save_hyperparameters()
                
        self.rce_kwargs = rce_kwargs 
        self.optimizer_kwargs = optimizer_kwargs
        
        self.RCE = RCE(**rce_kwargs)
        
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
    
    def forward(self, *args):
        return self.RCE(args)
    
    def training_step(self, batch, batch_idx):
        x =  self(batch.values())
        return None
    
    def validation_step(self, batch, batch_idx):
        x =  self(batch.values())
        return None
    
    def test_step(self, batch, batch_idx):
        x =  self(batch.values())
        return None