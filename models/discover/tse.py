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
        
        self.optimizer_kwargs = optimizer_kwargs
                
        self.gripper_up = nn.Linear(**gripper_up_kwargs)
        self.visionBackbone = VisionBackbone(**vision_backbone_kwargs)
        self.visionEncoder = VisionEncoder(**vision_encoder_kwargs)
        self.kmeans = KMeans(**kmeans_kwargs).eval()
        self.rgmm = RGMM(**rgmm_kwargs)
        
    def setup(self, stage):
        self.rgmm.logger = getattr(self.logger, "experiment", None)
        return None
    
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
        x: TensorType["b*ck*wd", "c", "h", "w"],
        x_shape: TensorType["b*ck*wd", "c", "h", "w"], 
        idxs: TensorType["b", "c", "wd"]
        ) -> TensorType["b*ck", "d_model"]:
        
        x = self.visionBackbone(x) # [b*ck*wd, c_model]
        x = x.view(-1, x_shape[2], x.shape[-1]) # [b*c, w, d_model]
        x = self.visionEncoder(x, idxs).squeeze() # [b*ck, d_model]

        return x 
 
    def _shared_step(self, batch: Dict[str, TensorType["b", "ck", "wd", "*"]], stage: str) -> float: 
        rgb_shape = batch["rgb_one"].shape # [b, ck, wd, c=3, h=224, w=224]
        
        rgb_one = batch["rgb_one"].view(-1, *rgb_shape[3:]) # [b*ck*wd, c=3, h=224, w=224]
        rgb_one_plus = batch["rgb_one_plus"].view(-1, *rgb_shape[3:]) # [b*ck*wd, c=3, h=224, w=224]


        # g_qpos = batch["g_qpos"].view(-1)[:, None] # [b*ck*wd, 1]

        x_emb = self(rgb_one, rgb_shape, batch["idxs"]) # [b*ck, d_model]
        x_plus_emb = self(rgb_one_plus, rgb_shape, batch["idxs"]) # [b*ck, d_model]
        # gripper_emb = self.gripper(gripper_qpos) # [b*ck, wd, d_model]
                
        weights, means, covs, _ = self.kmeans(x_emb).values()
        loss = self.rgmm(x_emb, x_plus_emb, weights, means, covs)
        
        return loss
        
    def training_step(self, batch, batch_idx) -> TensorType["batch"]:  
        return self._shared_step(batch=batch, stage="train")
        
    def validation_step(self, batch, batch_idx) -> TensorType["batch"]:  
        return self._shared_step(batch=batch, stage="validate")

    def test_step(self, batch, batch_idx) -> TensorType["batch"]:        
        return self._shared_step(batch=batch, stage="test")