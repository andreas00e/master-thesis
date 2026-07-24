import wandb
from typing import Dict

import torch
from torchtyping import TensorType

import lightning.pytorch as pl 

from .gmm import GMM 
from .visionBackbone import VisionBackbone
from .visionEncoder import VisionEncoder


class SkillDiscovery(pl.LightningModule): 
    def __init__(
        self, 
        vision_backbone_kwargs: Dict, 
        vision_encoder_kwargs: Dict, 
        gmm_kwargs: Dict,
        optimizer_kwargs: Dict, 
        ) -> None:
        
        super().__init__()
        
        self.save_hyperparameters() 
        
        self.vision_backbone_kwargs = vision_backbone_kwargs 
        self.vision_encoder_kwargs = vision_encoder_kwargs 
        self.gmm_kwargs = gmm_kwargs 
        self.optimizer_kwargs = optimizer_kwargs
                
        self.visionBackbone = VisionBackbone(**self.vision_backbone_kwargs)
        self.visionEncoder = VisionEncoder(**self.vision_encoder_kwargs)
        self.gmm = GMM(**self.gmm_kwargs)   

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), **self.optimizer_kwargs.optimizer)
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, **self.optimizer_kwargs.lr_scheduler)
        
        return {
            "optimizer": optimizer, 
            "scheduler": scheduler
        }
    
    
    def forward(self, x: TensorType[""]) -> TensorType["*"]:
        x = self.visionBackbone(x)
        x = self.visionEncoder(x) 

        return x 
    
    def _log_image(self, x: TensorType["*"], x_name: str) -> None: 
            self.logger.log_image(
                key = x_name, 
                image = [wandb.Image(img) for img in x.detach().cpu().numpy()]
            )
 
    def _shared_step(self, batch): 
        x, x_plus, _ = batch.values()
        
        x_emb = self(x)
        x_plus_emb = self(x_plus)
        loss = self.gmm(x_emb, x_plus_emb)

        return loss

    def training_step(self, batch, batch_idx): 
        loss = self._shared_step(batch)
        self.log_dict({
            "train_loss": loss
        })
        
    def validation_step(self, batch, batch_idx):
        loss = self._shared_step(batch)
        self.log_dict({
            "val_loss": loss
        })

    def test_step(self, batch, batch_idx): 
        loss = self._shared_step(batch)
        self.log_dict({
            "test_loss": loss
        })