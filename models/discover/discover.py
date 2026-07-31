import wandb
import numpy as np 
from typing import Dict

import torch
from torchtyping import TensorType
import lightning.pytorch as pl 

from .clustering.kmeans import KMeans
from .clustering.rgmm import GMM 
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
        self.kmeans = KMeans().eval()
        self.gmm = GMM(**self.gmm_kwargs)   

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
        x, x_plus = batch.values() 
        x_emb = self(x)  # [b, d_emb]
        # x_plus_emb = self(x_plus) # [b, d_emb]
        # loss = self.gmm(x_emb, x_plus_emb)
        
        stage = self.trainer.state.stage
        # self.log_dict({
        #     stage: loss
        #     }, 
        #     prog_bar=True,
        #     batch_size=x.shape
        # )
        
        if stage == "train":  
            with torch.no_grad(): 
                labels = self.kmeans(x_emb)
            
            labels = labels.detach().cpu().numpy()
            features = x_emb.detach().cpu().numpy()
            data = np.hstack((features, labels))
            columns = [f"feature_{i}" for i in range(x_emb.shape[0])] + ["label"]        
            
            table = wandb.Table(columns=columns, data=data)
            self.logger.experiment.log({"assignments": table})

        return None 
        
    def training_step(self, batch, batch_idx) -> TensorType["batch"]:  
        return self._shared_step(batch, "train")
        
    def validation_step(self, batch, batch_idx) -> TensorType["batch"]:  
        return self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx) -> TensorType["batch"]:        
        return self._shared_step(batch, "train")