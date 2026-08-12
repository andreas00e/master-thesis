# Skill Action Decoder (SAD)

import os 
from omegaconf import DictConfig

import torch
import torch.nn.functional as F 
from torchtyping import TensorType
import lightning.pytorch as pl

from models.discover.tse import TSE
from models.transfer.utils.sat import SAT
from models.transfer.utils.rce import RCE
from models.transfer.utils.ditbc import DiTBC


class SAD(pl.LightningModule): 
    def __init__(
        self, 
        ditbc_kwargs: DictConfig, 
        rce_kwargs: DictConfig,
        sat_kwargs: DictConfig, 
        optimizer_kwargs: DictConfig,
        tse_ckpt: os.PathLike, 
        ) -> None:
        
        super().__init__()
        self.save_hyperparameters()
       
        self.ditbc_kwargs = ditbc_kwargs
        self.rce_kwargs = rce_kwargs
        self.sat_kwargs = sat_kwargs
        self.optimizer_kwargs = optimizer_kwargs
        self.tse_ckpt = tse_ckpt # checkpoint of trained 
        
        # self.tse = TSE.load_from_checkpoint(self.tse_ckpt)
        self.tse = None
        
        
        # self.SAT = SAT(**self.sat_kwargs)
        # self.DiTBC = DiTBC(**self.ditbc_kwargs)
        self.RCE = RCE(**self.rce_kwargs)

        # self.loss = F.mse_loss()
    
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
    
    def forward(self, batch):
        actions, rgb_obs, joint_dsc, joint_obs = batch.values() 
        
        rce_emb = self.RCE(joint_dsc, joint_obs) # [batch, steps, d_model]
        

        sat_loss = 0 
        ditbc_loss = 0
        
        loss = sat_loss + ditbc_loss
        
        return loss 
    
    def _shared_step(self, batch: TensorType["batch"]) -> None: 
        loss = self(batch)
    
        return None
    
    def training_step(self, batch, batch_idx):
        return self._shared_step(batch)
    
    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch)
    
    def test_step(self, batch, batch_idx):
        return self._shared_step(batch)