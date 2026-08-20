# Temporal Skill Encoder (TSE)

from typing import Dict
from omegaconf import DictConfig

import torch
import torch.nn as nn 
from torchtyping import TensorType
import lightning.pytorch as pl 

from models.discover.utils.cluster import KMeans, RGMM
from models.discover.utils.visionBackbone import VisionBackbone
from models.discover.utils.visionEncoder import VisionEncoder


class TSE(pl.LightningModule): 
    def __init__(
        self, 
        gripper_up_kwargs: DictConfig, 
        vision_backbone_kwargs: DictConfig, 
        vision_encoder_kwargs: DictConfig, 
        kmeans_kwargs: DictConfig, 
        rgmm_kwargs: DictConfig,
        optimizer_kwargs: DictConfig, 
        ) -> None:
        
        super().__init__()
        self.save_hyperparameters() 
        
        self.gripper_up_kwargs = gripper_up_kwargs
        self.vision_backbone_kwargs = vision_backbone_kwargs 
        self.vision_encoder_kwargs = vision_encoder_kwargs 
        self.kmeans_kwargs = kmeans_kwargs
        self.rgmm_kwargs = rgmm_kwargs 
        self.optimizer_kwargs = optimizer_kwargs
                
        self.gripper_up = nn.Linear(**self.gripper_up_kwargs)
        self.visionBackbone = VisionBackbone(**self.vision_backbone_kwargs)
        self.visionEncoder = VisionEncoder(**self.vision_encoder_kwargs)
        self.kmeans = KMeans(**self.kmeans_kwargs)
        self.rgmm = RGMM(**self.rgmm_kwargs)
        self.kmeans.eval() 
        
    def setup(self, stage):
        if self.logger and hasattr(self.logger, "experiment"): 
            self.rgmm.logger = self.logger.experiment
        
        return super().setup(stage)
    
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
    
    def forward(
        self, 
        x: TensorType["batch*chunk*window", "channels", "height", "width"],
        x_shape: TensorType["batch*chunk*window", "channels", "height", "width"], 
        idxs: TensorType["batch", "chunk", "window"]
        ) -> TensorType["batch*chunk", "hidden_dim"]:
        
        x = self.visionBackbone(x) # [batch*chunk*window, hidden_dim]
        x = x.view(-1, x_shape[2], x.shape[-1]) # [batch*chunk, window, hidden_dim]
        x = self.visionEncoder(x, idxs).squeeze() # [batch*chunk, hidden_dim]

        return x 
 
    def _shared_step(self, batch: Dict[str, TensorType["batch", "chunk", "window", "*"]], stage: str) -> None: 
        rgb_one, rgb_one_plus, rgb_two, rgb_two_plus, g_qpos, g_qpos_plus, idxs = batch.values()
        
        rgb_shape = rgb_one.shape # [batch, chunk, window, channels, height, width]
        
        rgb_one = rgb_one.view(-1, *rgb_shape[3:]) # [batch*chunk*window, channels, height, width]
        rgb_one_plus = rgb_one_plus.view(-1, *rgb_shape[3:]) # [batch*chunk*window, channels, height, width]
        
        g_qpos = g_qpos.view(-1)[:, None] # [batch*chunk*window, 1]

        x_emb = self(rgb_one, rgb_shape, idxs) # [batch*chunk*window, hidden_dim]
        x_plus_emb = self(rgb_one_plus, rgb_shape, idxs) # # [batch*chunk*window, hidden_dim]
        # gripper_emb = self.gripper(gripper_qpos) # [batch*chunk, window, 256]
                
        weights, means, covs, _ = self.kmeans(x_emb).values()
        loss_cl, loss_bml = self.rgmm(x_emb, x_plus_emb, weights, means, covs)
        loss = loss_cl + self.rgmm_kwargs.bml_weight * loss_bml
        
        self.log_dict({
            f"{stage}_loss_cl": 
                loss_cl, 
            f"{stage}_loss_bml": 
                loss_bml, 
            f"{stage}_loss": 
                loss
        })
        
        return loss
        
    def training_step(self, batch, batch_idx) -> TensorType["batch"]:  
        return self._shared_step(batch=batch, stage="train")
        
    def validation_step(self, batch, batch_idx) -> TensorType["batch"]:  
        return self._shared_step(batch=batch, stage="validate")

    def test_step(self, batch, batch_idx) -> TensorType["batch"]:        
        return self._shared_step(batch=batch, stage="test")