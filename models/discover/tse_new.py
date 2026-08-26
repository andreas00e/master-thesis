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
        rgb_shape = batch["rgb_one"].shape # [batch, chunk, window, channels=3, height=224, width=224]
        
        
        rgb_one = batch["rgb_one"] # robot0_eye_in_hand_image
        rgb_two = batch["rgb_two"] # agentview_image
        idxs = batch["idxs"] # idxs
        
        rgb_one = rgb_one.view(-1, *rgb_shape[3:])
        rgb_two = rgb_two.view(-1, *rgb_shape[3:])
        
        one_emb = self(rgb_one, rgb_shape, idxs)
        two_emb = self(rgb_two, rgb_shape, idxs)
        
        one_emb = self.clusterHead(one_emb) # [n, k]
        two_emb = self.clusterHead(two_emb) # [n, k]
        
        y_one = self.distributed_sinkhorn(two_emb) # [n, k]
        y_two = self.distributed_sinkhorn(one_emb) # [n, k]
        
        y_hat_one = F.softmax(one_emb, dim=-1) # [n, k]
        y_hat_two = F.softmax(two_emb, dim=-1) # [n, k]
        
        loss_one = F.cross_entropy(y_one, y_hat_one)  
        loss_two = F.cross_entropy(y_two, y_hat_two)
        
        loss = 1/2 * (loss_one + loss_two)
        
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