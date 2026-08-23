from typing import List
from omegaconf import DictConfig

import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torchvision.models import resnet18
import lightning.pytorch as pl
from torchtyping import TensorType 

from models.discover.utils.visionEncoder import VisionEncoder
from models.discover.utils.cluster import KMeans
from models.discover.utils.vicreg import VICReg


class TSE(pl.LightningModule): 
    def __init__(
        self, 
        cluster_head_kwargs: DictConfig, 
        vision_encoder_kwargs: DictConfig, 
        vic_reg_kwargs: DictConfig, 
        kmeans_kwargs: DictConfig, 
        optimizer_kwargs: DictConfig, 
        ) -> None: 
        
        super().__init__()
        self.save_hyperparameters() 
        
        self.optimizer_kwargs = optimizer_kwargs
        
        self.clusterHead = nn.Linear(**cluster_head_kwargs)
        self.visionBackbone = resnet18(weights=None)
        self.visionEncoder = VisionEncoder(**vision_encoder_kwargs)
        self.vicReg = VICReg(**vic_reg_kwargs)
        self.kMeans = KMeans(**kmeans_kwargs).eval()
        
        self.labels = self.register_buffer("labels", torch.empty(size=(1, ), dtype=torch.int, device=self.device))
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")
        
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
        x: TensorType["batch*chunk*window", "d_model"], 
        x_shape: torch.Size, 
        idxs: TensorType["batch", "chunk", "window"]
        ) -> TensorType["batch*chunk", "d_model"]:
        
        x = self.visionBackbone(x) # [batch*chunk*window, d_model]
        x = x.view(*x_shape[:2], x_shape[2], -1) # [batch*chunk, window, d_model]
        x = self.visionEncoder(x, idxs) # [batch*chunk, 1, d_model]
        
        return x
    
    def _klDivLoss(self, x, y):
        x = F.log_softmax(x, dim=-1)
        y = F.softmax(y, dim=-1) 
        a = self.kl_loss(x, y)
        b = self.kl_loss(y, x)
        out = 0.5 * (a + b)
        
        return out 
    
    def _shared_step(self, batch, batch_idx, stage): 
        rgb_shape = batch["rgb_one"].shape # [batch, chunks, window, channels=3, height=224, width=224]
        
        rgb_one = batch["rgb_one"].view(-1, *rgb_shape[3:]) # [batch*chunk*window, channels=3, height=224, width=224]
        rgb_two = batch["rgb_two"].view(-1, *rgb_shape[3:]) # [batch*chunk*window, channels=3, height=224, width=224]
        idxs = batch["idxs"]
        
        rgb_one_emb = self(rgb_one, rgb_shape, idxs) # [batch*chunk, d_model]
        rgb_two_emb = self(rgb_two, rgb_shape, idxs) # [batch*chunk, d_model]
        
        loss_align = self.vicReg(rgb_one_emb, rgb_two_emb) # []
        
        embs = 0.5 * (rgb_one_emb + rgb_two_emb) # [batch*chunk, d_model]

        if batch_idx == 1:
            self.labels = self.labels.expand(rgb_one.shape[0]) # [batch*chunk]
            self.labels = self.kMeans(embs.detach()).unsqueeze(-1) # [batch*chunk, 1]
            
        x_emb = self.clusterHead(embs) # [batch*chunk, 1]
        
        loss_cluster = F.cross_entropy(x_emb, self.labels)
 
        logits_rgb_one = self.clusterHead(rgb_one_emb) # [batch*chunk, 1]
        logits_rgb_two = self.clusterHead(rgb_two_emb) # [batch*chunk, 1]
        
        kl_one = self._klDivLoss(logits_rgb_one, logits_rgb_two)
        
        return loss_align + loss_cluster + kl_one
    
    def training_step(self, batch, batch_idx) -> TensorType["batch"]:  
        return self._shared_step(batch=batch, batch_idx=batch_idx, stage="train")
    
    def validation_step(self, batch, batch_idx) -> TensorType["batch"]:  
        return self._shared_step(batch=batch, batch_idx=batch_idx, stage="validate")

    def test_step(self, batch, batch_idx) -> TensorType["batch"]:        
        return self._shared_step(batch=batch, batch_idx=batch_idx, stage="test")