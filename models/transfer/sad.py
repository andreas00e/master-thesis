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
from models.transfer.utils.dit import DIT


class SAD(pl.LightningModule): 
    def __init__(
        self, 
        rce_kwargs: DictConfig,
        dit_kwargs: DictConfig, 
        sat_kwargs: DictConfig, 
        optimizer_kwargs: DictConfig,
        tse_ckpt: os.PathLike=None, 
        ) -> None:
        super().__init__()
        self.save_hyperparameters()
       
        self.dit_kwargs = dit_kwargs
        self.rce_kwargs = rce_kwargs
        self.sat_kwargs = sat_kwargs
        self.optimizer_kwargs = optimizer_kwargs
        self.tse_ckpt = tse_ckpt 
        
        self.tse = None
        self.rce = RCE(**self.rce_kwargs)
        self.dit = DIT(**self.dit_kwargs)
        self.sat = SAT(**self.sat_kwargs)
 
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
        # rce_emb = self.rce(joint_dsc, joint_obs) # [batch, steps, d_model]
        # loss_sat = self.sat(rgb_obs)
        
        loss_bc = self.dit(actions, rgb_obs)
        
        # loss_bc = self.DiTBC(actions, rgb_obs)
        
        loss_bc = 0 
        loss_sat = 0
        loss = loss_sat + loss_bc

        stage = self.trainer.state.stage
        self.log_dict({
            f"{stage}_loss_sat": loss_sat, 
            f"{stage}_loss_bc": loss_bc, 
            f"{stage}_loss": loss
        })
        
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