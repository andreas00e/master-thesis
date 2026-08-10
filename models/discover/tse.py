# Temporal Skill Encoder (TSE)

from typing import Dict

import torch
from torchtyping import TensorType
import lightning.pytorch as pl 

from models.discover.cluster import KMeans, RGMM
from models.discover.visionBackbone import VisionBackbone
from models.discover.visionEncoder import VisionEncoder


class TSE(pl.LightningModule): 
    def __init__(
        self, 
        vision_backbone_kwargs: Dict, 
        vision_encoder_kwargs: Dict, 
        kmeans_kwargs: Dict, 
        rgmm_kwargs: Dict,
        optimizer_kwargs: Dict, 
        ) -> None:
        super().__init__()
        self.save_hyperparameters() 
        
        self.vision_backbone_kwargs = vision_backbone_kwargs 
        self.vision_encoder_kwargs = vision_encoder_kwargs 
        self.kmeans_kwargs = kmeans_kwargs
        self.rgmm_kwargs = rgmm_kwargs 
        self.optimizer_kwargs = optimizer_kwargs
                
        self.visionBackbone = VisionBackbone(**self.vision_backbone_kwargs)
        self.visionEncoder = VisionEncoder(**self.vision_encoder_kwargs)
        self.kmeans = KMeans(**self.kmeans_kwargs)
        self.rgmm = RGMM(**self.rgmm_kwargs)   
        
        self.kmeans.eval()

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
    
    def forward(self, x: TensorType["batch"]) -> TensorType["batch"]:
        x = self.visionBackbone(x)
        x = self.visionEncoder(x)

        return x 
 
    def _shared_step(self, batch: TensorType["batch"], stage: str) -> None: 
        x, x_plus, gripper_qpos = batch.values() 
        
        x = x.view(-1, *x.shape[3:]) # [batch*chunk*window, channels, height, width]
        x_plus = x_plus.view(-1, *x_plus.shape[3:]) # [batch*chunk*window, channels, height, width]
        gripper_qpos = gripper_qpos.view(-1)[:, None] # [batch*chunk*window, 1]
        
        x_emb = self(x) # [batch*chunk*window, hidden_dim]
        x_plus_emb = self(x_plus) # # [batch*chunk*window, hidden_dim]
        
        weights, means, covs, _ = self.kmeans(x).values()
        self.rgmm = RGMM(weights=weights, means=means, covs=covs, **self.gmm_kwargs).to(self.device)
        
        loss = self.rgmm(x_emb, x_plus_emb)
        return loss 
        
    def training_step(self, batch, batch_idx) -> TensorType["batch"]:  
        return self._shared_step(batch, "train")
        
    def validation_step(self, batch, batch_idx) -> TensorType["batch"]:  
        return self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx) -> TensorType["batch"]:        
        return self._shared_step(batch, "train")