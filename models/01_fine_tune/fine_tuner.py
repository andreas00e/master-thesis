from typing import Dict
from omegaconf import DictConfig
from hydra.utils import instantiate

import torch
import torch.nn as nn 

import lightning.pytorch as pl

from models.A1_utils.vision import VisionEncoder, Expander
from models.A1_utils.vicreg import VICReg
from models.A1_utils.loss import DynamicWeightAverage
from models.A1_utils.utils import SinusoidalEmbedding

class FineTuner(pl.LightningModule): 
    def __init__(
        self, 
        optimizer_kwargs: DictConfig, 
        scheduler_kwargs: DictConfig, 
        vision_encoder_kwargs: DictConfig, 
        gripper_encoder_kwargs: DictConfig,
        vic_reg_kwargs: DictConfig,
        dwa_kwargs: DictConfig
        ) -> None: 
        
        super().__init__()
        self.save_hyperparameters() 
        
        self.optimizer_kwargs = optimizer_kwargs
        self.scheduler_kwargs = scheduler_kwargs
        self.vision_encoder_kwargs = vision_encoder_kwargs, 
        self.gripper_encoder_kwargs = gripper_encoder_kwargs
        self.expander_kwargs = gripper_expander_kwargs
        self.vic_reg_kwargs = vic_reg_kwargs
        self.dwa_kwargs = dwa_kwargs
        
        self.visionEncoder = VisonEncoder(**self.vision_encoder_kwargs)
        self.visionExpander = Expander(**self.expander_kwargs)
        self.gripperEncoder = nn.Linear(**self.gripper_encoder_kwargs)
        self.gripperExpander = Expander(**self.expander_kwargs)
        
        self.vicReg = VICReg(**self.vic_reg_kwargs)
        self.dwa = DynamicWeightAverage(**self.dwa_kwargs)
        self.sinusoidalEmbedding = SinusoidalEmbedding(self.gripper_encoder_kwargs.out_features // 2)
        
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
        rgb_one_emb = self.visionEncoder(batch["rgb_one"]) 
        rgb_two_emb = self.visionEncoder(batch["rgb_two"]) 
        gripper_emb = self.gripperBackbone(batch["g_qpos"])
        
        loss_one, logs_one = self.vicReg(rgb_two_emb, rgb_one_emb) 
        loss_two, logs_two = self.vicReg(rgb_two_emb, gripper_emb)   
        losses = torch.vstack((loss_one, loss_two)) # [2, 1]

        lambda_ = self.dwa(self.losses)
        loss = torch.sum(lambda_ * losses)
        
        is_train = stage == "train"
        
        self.log_dict(
            {f"{stage}/loss_one/{k}": v for k, v in log_one.items()}, 
            prog_bar=True,
            on_step=is_train, 
            on_epoch=True, 
            sync_dist=True 
        )
        self.log_dict(
            {f"{stage}/loss_two/{k}": v for k, v in loss_two.items()}, 
            prog_bar=False,
            on_step=is_train, 
            on_epoch=True, 
            sync_dist=True 
        )
        self.log_dict({f"{stage}/loss": loss}, 
            prog_bar=True,
            on_step=is_train, 
            on_epoch=True, 
            sync_dist=True 
            )

        return loss 