import os 
from typing import Optional 
from omegaconf import DictConfig

import torch
import torch.nn.functional as F 
import lightning.pytorch as pl

from models.discover.tse import TSE
from models.transfer.utils.sat import SAT
from models.transfer.utils.rce import RCE
from models.transfer.utils.ditbc import DiTBC


class SkillTransfer(pl.LightningModule): 
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

        self.tse_ckpt = tse_ckpt
         
        # self.TSE = TSE.load_from_checkpoint(self.tse_ckpt)
        # self.ditbc_kwargs["device"] = self.device      
        # self.DiTBC = DiTBC(**self.ditbc_kwargs)
        self.RCE = RCE(**self.rce_kwargs)
        ## self.SAT = SAT(**self.sat_kwargs)

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
    
    def _shared_step(self, rgb_obs, actions, joint_dsc, joint_obs): 
        # T = rgb_obs.shape[0]
        # t = torch.randint(low=0, high=T).item()

        # z_tilde = self.TSE(rgb_obs)
        # z_t_hat = self.SAT(rgb_obs)
        # sat_loss = self.loss(z_t_hat, z_tilde[t])
        
        # prop_t = self.RCE(joint_dsc, joint_obs)
        # actions_hat = self.DiTBC(rgb_obs[t], prop_t, z_t_hat)
        # bc_loss = self.loss(actions_hat, actions)
        
        # loss = sat_loss + bc_loss 
        loss = 0 
        self.log_dict({f"loss_{self.trainer.state.stage}": loss})
    
        return loss
    
    def forward(self, batch):
        actions, rgb_obs, joint_dsc, joint_obs = batch.values() 
        loss = self._shared_step(actions, rgb_obs, joint_dsc, joint_obs)
        return loss
    
    def training_step(self, batch, batch_idx):
        x = self._shared_step(batch)
        return None
    
    def validation_step(self, batch, batch_idx):
        x = self._shared_step(batch)
        return None
    
    def test_step(self, batch, batch_idx):
        x = self._shared_step(batch)
        return None