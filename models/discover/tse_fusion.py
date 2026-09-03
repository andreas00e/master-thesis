import numpy as np 
from omegaconf import DictConfig

import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torchvision.models import resnet18
import lightning.pytorch as pl
from torchtyping import TensorType 

class TSE(pl.LightningModule): 
    def __init__(
        self, 
        general: DictConfig, 
        cluster_head_kwargs: DictConfig, 
        vision_encoder_kwargs: DictConfig, 
        vic_reg_kwargs: DictConfig, 
        kmeans_kwargs: DictConfig, 
        optimizer_kwargs: DictConfig, 
        ) -> None: 
        
        super().__init__()
        self.save_hyperparameters() 
        
        print(general)
        
        self.optimizer_kwargs = optimizer_kwargs
        
        self.visionBackbone = resnet18(weights=None)
        self.attention = nn.MultiheadAttention(embed_dim=general.d_model, num_heads=1, batch_first=True)

        self.token = nn.Parameter(data=torch.empty(size=(8, general.d_model), dtype=torch.float32, device=self.device))
        nn.init.xavier_uniform_(self.token)
        
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
        x = x.view(x_shape[0]*x_shape[1], x_shape[2], -1) # [batch*chunk, window, d_model]
        x = self.visionEncoder(x, idxs).squeze() # [batch*chunk, d_model]
        
        return x

    def _shared_step(self, batch, batch_idx, stage): 
        rgb_shape = batch["rgb_one"].shape # [batch, chunks, window, channels=3, height=224, width=224]
        n = rgb_shape[0] * rgb_shape[1] * rgb_shape[2]
        
        rgb_one = batch["rgb_one"].view(-1, *rgb_shape[3:]) # [n=batch*chunk*window, channels=3, height=224, width=224]
        rgb_two = batch["rgb_two"].view(-1, *rgb_shape[3:]) # [n=batch*chunk*window, channels=3, height=224, width=224]
        
        token_one = self.visionBackbone(rgb_one) # [n, d_model]
        token_two = self.visionBackbone(rgb_two) # [n, d_model]
        
        seq = torch.cat(tensors=(token_one, token_two, self.token), dim=0) # [n*2+m, d_model]
        
        attn_mask = torch.zeros(size=(seq.shape[0], seq.shape[0]), dtype=torch.float32, device=self.device) # [n*2+m, n*2+m]
        
        l = int(seq.shape[0] / n - 1)
        for i in range(l):       
                attn_mask[n*(1+i):n*(2+i), :l*n] =  torch.full(size=(n, n), fill_value=-np.inf, dtype=torch.float32, device=self.device)
                
                attn_mask[n*(1+i):n*(2+i), n*i:n*(1+i)] = torch.full(size=(n, n), fill_value=-np.inf, dtype=torch.float32, device=self.device)
                attn_mask[n*i:n*(1+i), n*(1+i):n*(2+i)] = torch.full(size=(n, n), fill_value=-np.inf, dtype=torch.float32, device=self.device)
            
        out, _ = self.attention(query=seq, key=seq, value=seq, attn_mask=attn_mask)
        loss = F.mse_loss(seq, out)
        
        return loss
    
    def training_step(self, batch, batch_idx) -> TensorType["batch"]:  
        return self._shared_step(batch=batch, batch_idx=batch_idx, stage="train")
    
    def validation_step(self, batch, batch_idx) -> TensorType["batch"]:  
        return self._shared_step(batch=batch, batch_idx=batch_idx, stage="validate")

    def test_step(self, batch, batch_idx) -> TensorType["batch"]:        
        return self._shared_step(batch=batch, batch_idx=batch_idx, stage="test")