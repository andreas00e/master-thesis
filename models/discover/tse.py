import wandb
import numpy as np 
import pandas as pd 
import seaborn as sns
from typing import Dict
from omegaconf import DictConfig
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt

import torch
import torch.nn as nn 
import torch.nn.functional as F

import lightning.pytorch as pl 
from torchtyping import TensorType

from models.discover.utils.vision import VisionBackbone, VisionEncoder
from models.discover.utils.selfsupervised.vicreg import VICReg


class TSE(pl.LightningModule): 
    def __init__(
        self, 
        vision_backbone_kwargs: DictConfig, 
        vision_encoder_kwargs: DictConfig,
        vic_reg_kwargs: DictConfig, 
        skill_head_kwargs: DictConfig,  
        tsne_kwargs: DictConfig, 
        sinkhorn_kwargs: DictConfig, 
        optimizer_kwargs: DictConfig, 
        ) -> None: 
        
        super().__init__()
        self.save_hyperparameters() 
        
        self.sinkhorn_kwargs = sinkhorn_kwargs        
        self.optimizer_kwargs = optimizer_kwargs

        self.visionBackbone = VisionBackbone(**vision_backbone_kwargs)
        self.visionEncoder = VisionEncoder(**vision_encoder_kwargs)
        self.vicReg = VICReg(**vic_reg_kwargs)
        self.skillHead = nn.Linear(**skill_head_kwargs) 
        
        self.tsne = TSNE(**tsne_kwargs)
        
    def configure_optimizers(self) -> None:
        if self.trainer.max_epochs is not None: 
            self.optimizer_kwargs.lr_scheduler.T_max = self.trainer.estimated_stepping_batches
        else:
            self.optimizer_kwargs.lr_scheduler.T_max = 100_000
            
        optimizer = torch.optim.Adam(self.parameters(), **self.optimizer_kwargs.optimizer)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **self.optimizer_kwargs.lr_scheduler)
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler, 
                "interval": "step"
            }
        }

    def training_step(self, batch, batch_idx) -> TensorType[""]:  
        return self._shared_step(batch=batch, batch_idx=batch_idx, stage="train")
        
    def validation_step(self, batch, batch_idx) -> TensorType[""]:  
        return self._shared_step(batch=batch, batch_idx=batch_idx, stage="val")

    def test_step(self, batch, batch_idx) -> TensorType[""]:        
        return self._shared_step(batch=batch, batch_idx=batch_idx, stage="test")
    
    def forward(
        self, 
        x: TensorType["batch*chunk*window", "channels", "height", "width"],
        idxs: TensorType["batch", "chunk", "window"]
        ) -> TensorType["batch*chunk", "d_model"]:
        x_shape = x.shape
        
        x = x.view(-1, *x_shape[3:]) # [batch*chunk*window, channels=3, height=224, width=224]
        x = self.visionBackbone(x) # [batch*chunk*window, d_model]
        x = x.view(x_shape[0]*x_shape[1], x_shape[2], -1) # [batch*chunk, window, d_model]
        x = self.visionEncoder(x, idxs)# [batch*chunk, d_model]
        x = x.view(-1, x.shape[-1])
        
        return x
    
    def _shared_step(
        self, 
        batch: Dict[str, TensorType["batch", "chunk", "window", "*"]],
        batch_idx: int, 
        stage: str
        ) -> TensorType[""]: 
        idxs = batch["idxs"] # idxs
        
        one_emb = self(batch["rgb_one_anc"], idxs) # [n=batch*chunk, d_model]
        two_emb = self(batch["rgb_two_anc"], idxs) # [n=batch*chunk, d_model]
                
        one_emb = self.skillHead(one_emb)  # [n, k]
        two_emb = self.skillHead(two_emb) # [n, k]
        
        alignment_loss = self.vicReg(one_emb, two_emb) # align embedding spaces 

        target_one = self.distributed_sinkhorn(two_emb) # [n, k]
        target_two = self.distributed_sinkhorn(one_emb) # [n, k]
        
        prediction_one = F.log_softmax(one_emb / self.sinkhorn_kwargs.tau, dim=-1) # [n, k]
        prediction_two = F.log_softmax(two_emb / self.sinkhorn_kwargs.tau, dim=-1) # [n, k]
        
        loss_one = F.kl_div(prediction_one, target_one, reduction="batchmean")        
        loss_two = F.kl_div(prediction_two, target_two, reduction="batchmean")  
        prediction_loss = 1/2 * (loss_one + loss_two) 
            
        loss = alignment_loss + prediction_loss
        
        self.log_dict(
                {f"{stage}_prediction_loss": prediction_loss, 
                f"{stage}_alignment_loss": alignment_loss, 
                f"{stage}_loss": loss}
            )
        
        if batch_idx == 0 and stage == "val" and self.current_epoch % 5 == 0:
            self.plot_(one_emb, target_one, task=batch["task"], robot=batch["robot"])
        
        return loss 
    
    def plot_(
        self, 
        x: TensorType["n", "k"],
        label: TensorType["n"], 
        task: TensorType["n"], 
        robot: TensorType["n"]
        ) -> None:
        
        columns = ["x", "y", "label", "task", "robot"]
        
        x = x.clone().detach().cpu().numpy() # [n, k]
        label = label.argmax(-1).clone().detach().cpu().numpy() # [n]
        task = task.clone().detach().cpu().numpy() # [n]
        robot = robot.clone().detach().cpu().numpy() # [n]
        
        x = self.tsne.fit_transform(x) # [n, 2]
        data = np.stack(arrays=[x[:, 0], x[:, 1], label, task, robot], axis=-1)
        df = pd.DataFrame(data=data, columns=columns)
         
        plt.figure(figsize=(8, 6))
        scatterplot = sns.scatterplot(data=df, x="x", y="y", hue="label")
        fig = scatterplot.get_figure() 
        fig.savefig("scatterplot.png")
        
        self.log_dict({"tsne_plot": wandb.Image("scatterplot.png")})
        
        plt.close()
         
    @torch.no_grad()
    def distributed_sinkhorn(self, out):
    # Adjusted from https://github.com/real-stanford/xskill/blob/main/xskill/model/core.py
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