from typing import Dict
from omegaconf import DictConfig
from hydra.utils import instantiate

import torch
import torch.nn as nn 

import lightning.pytorch as pl

from models.discover.utils.models.vision import VisionBackbone, Encoder
from models.discover.utils.selfsupervised.vicreg import VICReg
from models.utils.loss import DynamicWeightAverage

class Pretrain(pl.LightningModule): 
    def __init__(
        self, 
        optimizer_kwargs: DictConfig, 
        scheduler_kwargs: DictConfig, 
        vision_backbone_kwargs: DictConfig, 
        vision_encoder_kwargs: DictConfig, 
        gripper_backbone_kwargs: DictConfig,
        gripper_encoder_kwargs: DictConfig, 
        vic_reg_kwargs: DictConfig,
        dwa_kwargs: DictConfig
        ) -> None: 
        
        super().__init__()
        self.save_hyperparameters() 
        
        self.optimizer_kwargs = optimizer_kwargs
        self.scheduler_kwargs = scheduler_kwargs
        
        self.visionBackbone = VisionBackbone(**vision_backbone_kwargs)
        self.visionEncoder = Encoder(**vision_encoder_kwargs)
        self.gripperBackbone = nn.Linear(**gripper_backbone_kwargs)
        self.gripperEncoder = Encoder(**gripper_encoder_kwargs)
        self.vicReg = VICReg(**vic_reg_kwargs)
        self.dwa = DynamicWeightAverage(**dwa_kwargs)
        
        
        self.register_parameter("losses", torch.ones(size=(2, self.dwa_kwargs.n_losses), dtype=torch.float32, device=self.device))        
        
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
        rgb_one_emb = self.visionEncoder(self.visionBackbone(batch["rgb_one"])) # [batch*chunk, d_model]
        rgb_two_emb = self.visionEncoder(self.visionBackbone(batch["rgb_two"])) # [batch*chunk, d_model]
        gripper_emb = self.gripperBackbone(batch["g_qpos"]) # [batch, chunk, window, d_model]
        gripper_emb = self.gripperEncoder(gripper_emb.view(-1, *gripper_emb.shape[-2:])) # [batch*chunk, d_model]
        
        loss_one = self.vicReg(rgb_one_emb, rgb_two_emb) 
        loss_two = self.vicReg(rgb_one_emb, gripper_emb)   
        
        lambda_ = self.dwa(self.losses)
        losses = torch.vstack((loss_one, loss_two)) # [2, 1]

        loss = torch.sum(lambda_ * losses)
        
        self.log_dict({
            f"{stage}_loss_one": loss_one, 
            f"{stage}_loss_two": loss_two, 
            f"{stage}_loss": loss
        })
        
        return loss 