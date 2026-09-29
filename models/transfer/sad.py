# Skill conditioned Action Decoder (SAD)

import os 
from omegaconf import DictConfig

from r3m import load_r3m

import torch
import torch.nn as nn
from torchtyping import TensorType
import lightning.pytorch as pl

from models.transfer.utils.sat import SAT
from models.transfer.utils.rce import RCE
from models.transfer.utils.dit import DIT
from models.transfer.utils.pooling import CrossAttentionQueryPooling
from models.discover.skill_encoder import SkillEncoder


class SAD(pl.LightningModule): 
    def __init__(
        self, 
        d_model: int, 
        tse_ckpt: os.PathLike,
        rce_kwargs: DictConfig,
        dit_kwargs: DictConfig, 
        pool_kwargs: DictConfig, 
        sat_kwargs: DictConfig, 
        optimizer_kwargs: DictConfig
        ) -> None:
        super().__init__()
        self.save_hyperparameters()
       
        self.d_model = d_model
        self.tse_ckpt = tse_ckpt 
        self.rce_kwargs = rce_kwargs
        self.dit_kwargs = dit_kwargs
        self.pool_kwargs = pool_kwargs
        self.sat_kwargs = sat_kwargs
        self.optimizer_kwargs = optimizer_kwargs
        
        # Frozen models
        self.obs_encoder = load_r3m("resnet18")
        self.tse = SkillEncoder.load_from_checkpoint(self.tse_ckpt)
        self.tse.eval() 
        self.tse.freeze()
        
        # Trainable models 
        self.obs_down = nn.Linear(
            512, 
            self.d_model
            )
        
        self.rce = RCE(**self.rce_kwargs)
        self.dit = DIT(**self.dit_kwargs)
        self.sat = SAT(self.tse, **self.sat_kwargs)
        self.pool = CrossAttentionQueryPooling(**self.pool_kwargs)

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
        conditions = []
        actions, rgb_obs, joint_dsc, joint_obs = batch.values() 
        
        
        batch_size, n_steps = actions.shape[:2]        
        item = {}
        item["rgb_one"] = rgb_obs.unsqueeze(0)
        item["rgb_one_pos"] = rgb_obs.unsqueeze(0)
        
        
        loss_sat = self.sat(item)
        
        rgb_obs = batch["rgb_obs"].view(-1, *rgb_obs.shape[2:]) # [batch*steps, channels=3, height=224, width=224]
        rgb_emb = self.obs_encoder(rgb_obs) # [batch*steps, *]
        rgb_emb = self.obs_down(rgb_emb) # [batch*steps, d_model]
        rgb_emb = rgb_emb.view(batch_size, n_steps, -1) # [batch, steps, d_model]
        
        rce_emb = self.rce(joint_dsc, joint_obs) # [batch, steps, d_model]
        
        skl_emb = torch.randn_like(rce_emb) # [batch, steps, d_model]: skill token prototypes 
        
        conditions = [rgb_emb, rce_emb, skl_emb] # list of conditions,,0
        conditions = self.pool(conditions) # [batch, n_steps, k, d_model] 
        
        loss_bc = self.dit(batch["actions"], conditions)       
         
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