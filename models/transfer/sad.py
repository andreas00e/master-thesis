import os 
import numpy as np
from typing import Any, Dict
from omegaconf import DictConfig

import torch
import torch.nn as nn
import lightning.pytorch as pl

from models.transfer.utils.dit import DiffusionTransformer
from models.transfer.utils.pooling import CrossAttentionQueryPooling
from models.transfer.utils.rce import RobotConditionedEncoder
from models.transfer.utils.sat import SkillAllignmentTransformer

from models.discover.skill_encoder import SkillEncoder
from models.utils.aux_models import CNN


class SkillConditionedActionDecoder(pl.LightningModule): 
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
        
        # Frozen model          
        self.tse = SkillEncoder.load_from_checkpoint(self.tse_ckpt)
        self.tse.requires_grad_(False)
        self.tse.eval()
        
        # Trainable models 
        self.obs_encoder = CNN(self.d_model)
        
        self.gripperEncoder = nn.Sequential(
            nn.Linear(1, self.d_model*2), 
            nn.ReLU(), 
            nn.Linear(self.d_model*2, self.d_model)
        )  
        
        self.rce = RobotConditionedEncoder(**self.rce_kwargs)
        self.dit = DiffusionTransformer(**self.dit_kwargs)
        self.sat = SkillAllignmentTransformer(**self.sat_kwargs)
        self.attention_pooling = CrossAttentionQueryPooling(**self.pool_kwargs)

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
    
    def forward(self, batch: Dict[str, Any], stage: str) -> torch.Tensor:
        conditions = []
        
        batch_size, num_steps, condition_horizon, = batch["rgb_one"].shape[:3]
        
        z_tilde = self.tse.predict_step(batch) # [batch*num_steps, k]
        z_tilde = z_tilde.view(batch_size, num_steps, -1) # [batch, num_steps, k]
                
        rgb_one = batch["rgb_one"].view(-1, *batch["rgb_one"].shape[3:]) # [batch*num_steps*condition_horizon, channels=3, height=224, width=224]
        rgb_one = self.obs_encoder(rgb_one) # [batch*num_steps*condition_horizon, d_model]
        rgb_one = rgb_one.view(batch_size, num_steps, condition_horizon, -1) # [batch, num_steps, condition_horizon, d_model]
        
        rgb_two = batch["rgb_two"].view(-1, *batch["rgb_two"].shape[3:]) # [batch*num_steps*condition_horizon, channels=3, height=224, width=224]
        rgb_two = self.obs_encoder(rgb_two) # [batch*num_steps*condition_horizon, d_model]
        rgb_two = rgb_two.view(batch_size, num_steps, condition_horizon, -1) # [batch, num_steps, condition_horizon, d_model]
        
        rce_emb = self.rce(batch["joint_dsc"], batch["joint_obs"]) # [batch, num_steps, condition_horizon, d_model]
        gripper_emb = self.gripperEncoder(batch["g_qpos"]) # [batch, num_steps, condition_horizon, d_model]
        
        conditions = [rgb_one, rgb_two, rce_emb, gripper_emb] 
        conditions = self.attention_pooling(conditions) # [batch, num_steps, condition_horizon, m, d_model] 
    
        loss_sat = self.sat(z_tilde, conditions) # []
        loss_bc = self.dit(batch["actions"], torch.mean(conditions, dim=1), batch["actions_idxs"], batch["conditions_idxs"][:, 0, ...])         
        loss = loss_bc + loss_sat
        
        self.log_dict({
            f"{stage}/loss_sat": loss_sat,
            f"{stage}/loss_bc": loss_bc,
            f"{stage}/loss": loss,

        },                            
        logger=True, 
        prog_bar=True,
        on_step=(stage == "train"), 
        on_epoch=True,
        sync_dist=True,
        )
        
        return loss 
    
    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:  
        return self(batch, stage="train")
        
    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:  
        return self(batch, stage="val")