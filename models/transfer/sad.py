import os 
from typing import Any
from omegaconf import DictConfig

from r3m import load_r3m

import torch
torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import lightning.pytorch as pl

from models.transfer.utils.sat import SkillAllignmentTransformer
from models.transfer.utils.rce import RobotConditionedEncoder
from models.transfer.utils.dit import DiffusionTransformer
from models.transfer.utils.pooling import CrossAttentionQueryPooling
from models.discover.skill_encoder import SkillEncoder


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
        
        # Frozen models
        self.obs_encoder = load_r3m("resnet18")
        self.obs_encoder.requires_grad_(False)
        self.obs_encoder.eval()
          
        self.tse = SkillEncoder.load_from_checkpoint(self.tse_ckpt)
        self.tse.requires_grad_(False)
        self.tse.eval() 
        
        # Trainable models 
        self.obs_down = nn.Linear(512, self.d_model)
        
        self.rce = RobotConditionedEncoder(**self.rce_kwargs)
        self.dit = DiffusionTransformer(**self.dit_kwargs)
        self.sat = SkillAllignmentTransformer(self.obs_encoder, self.tse,  **self.sat_kwargs)
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
    
    def forward(self, batch: Any, batch_idx: int, stage: str) -> torch.Tensor:
        conditions = []
        
        batch_size, seq_len = batch["rgb_one"].shape[:2]
                
        rgb_one = batch["rgb_one"].view(-1, *batch["rgb_one"].shape[2:]) # [batch*seq_len, channels=3, height=224, width=224]
        rgb_one = self.obs_encoder(rgb_one) # [batch*seq_len, *]
        rgb_one = self.obs_down(rgb_one) # [batch*seq_len, d_model]
        rgb_one = rgb_one.view(batch_size, seq_len, -1) # [batch, seq_len, d_model]
        
        rgb_two = batch["rgb_two"].view(-1, *batch["rgb_two"].shape[2:]) # [batch*seq_len, channels=3, height=224, width=224]
        rgb_two = self.obs_encoder(rgb_two) # [batch*seq_len, *]
        rgb_two = self.obs_down(rgb_two) # [batch*seq_len, d_model]
        rgb_two = rgb_two.view(batch_size, seq_len, -1) # [batch, seq_len, d_model]
        
        rce_emb = self.rce(batch["joint_dsc"], batch["joint_obs"]) # [batch, seq_len, d_model]

        conditions = [rgb_one, rgb_two, rce_emb] 
        conditions = self.attention_pooling(conditions) # [batch, seq_len, k, d_model] 
        
        loss_bc = self.dit(batch["actions"], conditions)       

        self.log_dict({
            f"{stage}/loss_bc": loss_bc, 
        },                             
        logger=True, 
        prog_bar=True,
        on_step=(stage == "train"), 
        on_epoch=True,
        sync_dist=True,
        )
        
        return loss_bc 
    
    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:  
        return self(batch, batch_idx, stage="train")
        
    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:  
        return self(batch, batch_idx, stage="val")