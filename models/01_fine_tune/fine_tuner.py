from typing import Dict
from omegaconf import DictConfig
from hydra.utils import instantiate

import torch
import torch.nn as nn 

import lightning.pytorch as pl

from models.A1_utils.vision import VisionBackbone
from models.A1_utils.vicreg import VICReg
from models.A1_utils.loss import DynamicWeightAverage

class FineTuner(pl.LightningModule): 
    def __init__(
        self, 
        optimizer_kwargs: DictConfig, 
        scheduler_kwargs: DictConfig, 
        vision_backbone_kwargs: DictConfig, 
        gripper_backbone_kwargs: DictConfig,
        vic_reg_kwargs: DictConfig,
        dwa_kwargs: DictConfig
        ) -> None: 
        
        super().__init__()
        self.save_hyperparameters() 
        
        self.optimizer_kwargs = optimizer_kwargs
        self.scheduler_kwargs = scheduler_kwargs
        self.vision_backbone_kwargs = vision_backbone_kwargs
        self.gripper_backbone_kwargs = gripper_backbone_kwargs
        self.vic_reg_kwargs = vic_reg_kwargs
        self.dwa_kwargs = dwa_kwargs
        
        self.visionBackbone = VisionBackbone(**self.vision_backbone_kwargs)
        self.gripperBackbone = nn.Sequential(
            nn.Flatten(start_dim=0, end_dim=-2),
            nn.Linear(**self.gripper_backbone_kwargs)
        )
        self.vicReg = VICReg(**self.vic_reg_kwargs)
        self.dwa = DynamicWeightAverage(**self.dwa_kwargs)
        
        self.register_buffer("losses", torch.ones(size=(2, self.dwa_kwargs.n_losses), dtype=torch.float32, device=self.device)) # XXX: '2' could be move to config
        
    def configure_optimizers(self) -> Dict:
        optimizer = instantiate(self.optimizer_kwargs, params=self.parameters())
        scheduler = instantiate(self.scheduler_kwargs, optimizer=optimizer)
        scheduler.T_max = self.trainer.estimated_stepping_batches
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler, 
                "interval": "step"
            }
        }
            
    def training_step(self, batch, batch_idx) -> torch.Tensor:
        return self(batch, batch_idx, stage="train")
    
    def validation_step(self, batch, batch_idx) -> torch.Tensor:
        return self(batch, batch_idx, stage="val")
    
    def test_step(self, batch, batch_idx) -> torch.Tensor:
        return self(batch, batch_idx, stage="test")
    
    def forward(self, batch, batch_idx, stage) -> torch.Tensor: 
        rgb_one_emb = self.visionBackbone(batch["rgb_one"]) 
        rgb_two_emb = self.visionBackbone(batch["rgb_two"]) 
        gripper_emb = self.gripperBackbone(batch["g_qpos"])
        
        loss_one = self.vicReg(rgb_one_emb, rgb_two_emb) 
        loss_two = self.vicReg(rgb_one_emb, gripper_emb)   
        
        lambda_ = self.dwa(self.losses)
        losses = torch.vstack((loss_one, loss_two)) # [2, 1]

        loss = torch.sum(lambda_ * losses)
        
        self.log_dict({
            f"{stage}_loss_one": loss_one, 
            f"{stage}_loss_two": loss_two, 
            f"{stage}_loss": loss
            }, 
            prog_bar=True,
            on_step=True, 
            on_epoch=True, 
            sync_dist=True 
        )

        return loss 