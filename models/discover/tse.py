from typing import Dict
from omegaconf import DictConfig
from torchvision.models import squeezenet1_1

import torch
import torch.nn as nn 
import torch.nn.functional as F
from torchtyping import TensorType
import lightning.pytorch as pl 

from models.discover.utils.cluster import KMeans, RGMM
from models.discover.utils.visionBackbone import VisionBackbone
from models.discover.utils.visionEncoder import VisionEncoder


class TSE(pl.LightningModule): 
    def __init__(
        self,
        sinkhorn_iterations: int, 
        vision_encoder_kwargs: DictConfig, 
        cluster_head_kwargs: DictConfig, 
        optimizer_kwargs: DictConfig, 
        ) -> None: 
        
        super().__init__()
        self.save_hyperparameters() 
                
        self.sinkhorn_iterations = sinkhorn_iterations
        self.optimizer_kwargs = optimizer_kwargs
                
        self.visionEncoder = VisionEncoder(**vision_encoder_kwargs)
        self.clusterHead = nn.Linear(**cluster_head_kwargs) 
        
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
        obs_shape = batch["rgb_one"].shape # [batch, chunk, window, channels=3, height=224, width=224]

        obs = batch["rgb_two"].view(-1, *obs_shape[3:]) # [batch*chunk*window, channels=3, height=224, width=224]
        obs_plus = batch["rgb_two_plus"].view(-1, *obs_shape[3:]) # [batch*chunk*window, channels=3, height=224, width=224]
        
        obs_emb = self(obs, obs_plus, batch["idxs"]) # [n=batch*chunk, d_model]
        obs_plus_emb = self(obs, obs_plus, batch["idxs"]) # [n=batch*chunk, d_model]
                
        obs_emb = self.clusterHead(obs_emb) # [n, k]
        obs_plus_emb = self.clusterHead(obs_emb) # [n, k]
        
        p_A = F.softmax(obs, dim=-1) # [n, k]
        p_B = F.softmax(obs, dim=-1) # [n, k]
        
        s_A = self.distributed_sinkhorn(obs_emb) # [n, k]
        s_B = self.distributed_sinkhorn(obs_plus_emb) # [n, k]
        
        
        loss_a = F.cross_entropy(p_A, s_B)  
        loss_b = F.cross_entropy(p_B, s_A)
        
        loss = 1/2 * (loss_a + loss_b)
        
        return loss
        
    def training_step(self, batch, batch_idx) -> TensorType["batch"]:  
        return self._shared_step(batch=batch, stage="train")
        
    def validation_step(self, batch, batch_idx) -> TensorType["batch"]:  
        return self._shared_step(batch=batch, stage="validate")

    def test_step(self, batch, batch_idx) -> TensorType["batch"]:        
        return self._shared_step(batch=batch, stage="test")
    
    @torch.no_grad()
    def distributed_sinkhorn(self, out):
        Q = torch.exp(out / self.epsilon).t(
        )  # Q is K-by-B for consistency with notations from our paper
        B = Q.shape[1]  # number of samples to assign
        K = Q.shape[0]  # how many prototypes

        # make the matrix sums to 1
        sum_Q = torch.sum(Q)
        Q /= sum_Q

        for _ in range(self.sinkhorn_iterations):
            # normalize each row: total weight per prototype must be 1/K
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
            Q /= sum_of_rows
            Q /= K

            # normalize each column: total weight per sample must be 1/B
            Q /= torch.sum(Q, dim=0, keepdim=True)
            Q /= B

        Q *= B  # the colomns must sum to 1 so that Q is an assignment
        return Q.t()