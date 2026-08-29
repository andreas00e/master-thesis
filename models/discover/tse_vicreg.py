from typing import Dict
from omegaconf import DictConfig
from torchvision import models

import torch
import torch.nn as nn 
import torch.nn.functional as F
from torchtyping import TensorType
import lightning.pytorch as pl 

from models.discover.utils.visionEncoder import VisionEncoder
from models.discover.utils.vicreg import VICReg


class TSE(pl.LightningModule): 
    def __init__(
        self, 
        vision_encoder_kwargs: DictConfig,
        vic_reg_kwargs: DictConfig, 
        cluster_head_kwargs: DictConfig,  
        sinkhorn_kwargs: DictConfig, 
        optimizer_kwargs: DictConfig, 
        ) -> None: 
        
        super().__init__()
        self.save_hyperparameters() 
        
        self.sinkhorn_kwargs = sinkhorn_kwargs        
        self.optimizer_kwargs = optimizer_kwargs
        
        self.visionBackbone = models.mobilenet_v3_small()
        self.visionEncoder = VisionEncoder(**vision_encoder_kwargs)
        self.vicReg = VICReg(**vic_reg_kwargs)
        self.skillHead = nn.Linear(**cluster_head_kwargs) 
        
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
        x_shape: torch.Size, 
        idxs: TensorType["batch", "chunk", "window"]
        ) -> TensorType["batch*chunk", "d_model"]:
        
        x = self.visionBackbone(x) # [batch*chunk*window, d_model]
        x = x.view(x_shape[0]*x_shape[1], x_shape[2], -1) # [batch*chunk, window, d_model]
        x = self.visionEncoder(x, idxs)# [batch*chunk, d_model]
        x = x.view(-1, x.shape[-1])
        
        return x
    
    def _shared_step(self, batch: Dict[str, TensorType["batch", "chunk", "window", "*"]], stage: str) -> TensorType[""]: 
        rgb_one = batch["rgb_one_anc"] # robot0_eye_in_hand_image
        rgb_two = batch["rgb_two_anc"] # agentview_image
        idxs = batch["idxs"]           # idxs
        
        rgb_shape = rgb_one.shape
        
        rgb_one = rgb_one.view(-1, *rgb_shape[3:]) # [batch*chunk*window, channels=3, height=224, width=224]
        rgb_two = rgb_two.view(-1, *rgb_shape[3:]) # [batch*chunk*window, channels=3, height=224, width=224]
        
        one_emb = self(rgb_one, rgb_shape, idxs) # [n=batch*chunk, d_model]
        two_emb = self(rgb_two, rgb_shape, idxs) # [n=batch*chunk, d_model]
                
        one_emb = self.skillHead(one_emb)  # [n, k]
        two_emb = self.skillHead(two_emb) # [n, k]
        
        alignment_loss = self.vicReg(one_emb, two_emb) # align embedding spaces 

        label_one = self.distributed_sinkhorn(two_emb) # [n, k]
        label_two = self.distributed_sinkhorn(one_emb) # [n, k]
        
        prediction_one = F.log_softmax(one_emb, dim=-1) # [n, k]
        prediction_two = F.log_softmax(two_emb, dim=-1) # [n, k]
        
        loss_one = F.kl_div(label_one, prediction_one, reduction="batchmean")
        loss_two = F.kl_div(label_two, prediction_two, reduction="batchmean")
        
        prediction_loss = 1/2 * (loss_one + loss_two) 
        loss = alignment_loss + prediction_loss
        
        self.log_dict(
            {f"{stage}_prediction_loss": prediction_loss, 
            f"{stage}_alignment_loss": alignment_loss, 
            f"{stage}_loss": loss}
        )
        
        return loss 
        
    def training_step(self, batch, batch_idx) -> TensorType["batch"]:  
        return self._shared_step(batch=batch, stage="train")
        
    def validation_step(self, batch, batch_idx) -> TensorType["batch"]:  
        return self._shared_step(batch=batch, stage="val")

    def test_step(self, batch, batch_idx) -> TensorType["batch"]:        
        return self._shared_step(batch=batch, stage="test")
    
    @torch.no_grad()
    def distributed_sinkhorn(self, out):
        Q = torch.exp(out / self.sinkhorn_kwargs.epsilon).t(
        )  # Q is K-by-B for consistency with notations from our paper
        B = Q.shape[1]  # number of samples to assign
        K = Q.shape[0]  # how many prototypes

        # make the matrix sums to 1
        sum_Q = torch.sum(Q)
        Q /= sum_Q

        for _ in range(self.sinkhorn_kwargs.sinkhorn_iterations):
            # normalize each row: total weight per prototype must be 1/K
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
            Q /= sum_of_rows
            Q /= K

            # normalize each column: total weight per sample must be 1/B
            Q /= torch.sum(Q, dim=0, keepdim=True)
            Q /= B

        Q *= B  # the colomns must sum to 1 so that Q is an assignment
        return Q.t()