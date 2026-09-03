import os 
import wandb
import tempfile
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
from models.discover.utils.queue import FIFOQueue

class TSE(pl.LightningModule): 
    def __init__(
        self, 
        queue_kwargs: DictConfig, 
        vision_backbone_kwargs: DictConfig, 
        vision_encoder_kwargs: DictConfig,
        vic_reg_kwargs: DictConfig, 
        prototype_kwargs: DictConfig,  
        tsne_kwargs: DictConfig, 
        sinkhorn_kwargs: DictConfig, 
        optimizer_kwargs: DictConfig, 
        ) -> None: 
        
        super().__init__()
        self.save_hyperparameters() 
        
        self.sinkhorn_kwargs = sinkhorn_kwargs        
        self.optimizer_kwargs = optimizer_kwargs
        
        self.queue = FIFOQueue(**queue_kwargs)
        self.visionBackbone = VisionBackbone(**vision_backbone_kwargs)
        self.visionEncoder = VisionEncoder(**vision_encoder_kwargs)
        self.vicReg = VICReg(**vic_reg_kwargs)
        self.C = nn.Linear(**prototype_kwargs) 
        nn.init.orthogonal_(self.C.weight)

        self.tsne = TSNE(**tsne_kwargs)
        
    def configure_optimizers(self) -> Dict:
        if self.trainer.max_epochs is not None: 
            self.optimizer_kwargs.lr_scheduler.two.T_max = self.trainer.estimated_stepping_batches
        else:
            self.optimizer_kwargs.lr_scheduler.two.T_max = 100_000
            
        optimizer = torch.optim.Adam(self.parameters(), **self.optimizer_kwargs.optimizer)
        scheduler_one = torch.optim.lr_scheduler.LinearLR(optimizer, **self.optimizer_kwargs.lr_scheduler.one)
        scheduler_two = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **self.optimizer_kwargs.lr_scheduler.two)
        scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[scheduler_one, scheduler_two], **self.optimizer_kwargs.lr_scheduler.sequential)
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler, 
                "interval": "step"
            }
        }

    def training_step(self, batch, batch_idx) -> torch.Tensor:  
        return self._shared_step(batch=batch, batch_idx=batch_idx, stage="train")
        
    def validation_step(self, batch, batch_idx) -> torch.Tensor:  
        return self._shared_step(batch=batch, batch_idx=batch_idx, stage="val")

    def test_step(self, batch, batch_idx) -> torch.Tensor:        
        return self._shared_step(batch=batch, batch_idx=batch_idx, stage="test")
    
    def forward(
        self, 
        x: TensorType["batch*chunk*window", "channels", "height", "width"],
        idxs: TensorType["batch", "chunk", "window"]
        ) -> TensorType["batch*chunk", "d_model"]:
        bs, ck, wd, c, h, w = x.shape
        
        x = x.view(-1, c, h, w) # [batch*chunk*window, channels=3, height=224, width=224]
        x = self.visionBackbone(x) # [batch*chunk*window, d_model]
        x = x.view(bs*ck, wd, -1) # [batch*chunk, window, d_model]
        x = self.visionEncoder(x, idxs)# [batch*chunk, d_model]
        
        return x
    
    def _shared_step(
        self, 
        batch: Dict[str, TensorType["batch", "chunk", "window", "*"]],
        batch_idx: int, 
        stage: str
        ) -> torch.Tensor: 
        idxs = batch["idxs"] # idxs
        
        h_one_batch = F.normalize(self(batch["rgb_one_anc"], idxs), dim=-1) # [n=batch*chunk, d_model]
        h_two_batch = F.normalize(self(batch["rgb_two_anc"], idxs), dim=-1) # [n=batch*chunk, d_model]
        
        n = h_one_batch.shape[0] 
       
        if self.queue is not None and self.queue.is_full: 
                queue_features = self.queue.dequeue() 
                h_one_assign = torch.cat([h_one_batch, queue_features[0]], dim=0)
                h_two_assign = torch.cat([h_two_batch, queue_features[1]], dim=0)
        else: 
            h_one_assign = h_one_batch
            h_two_assign = h_two_batch
            
        if self.queue is not None:   
            self.queue.enqueue(torch.stack([h_one_batch, h_two_batch], dim=0))
   
        # Map to prototypes
        z_one = self.C(h_one_assign) # [n, k]
        z_two = self.C(h_two_assign) # [n, k]
        
        # Find pseudo-labels
        with torch.no_grad(): 
            target_one = self.distributed_sinkhorn(z_two) # [n, k]
            target_two = self.distributed_sinkhorn(z_one) # [n, k]
        
        loss_one = F.cross_entropy(z_one[:n] / self.sinkhorn_kwargs.tau, target_one[:n])
        loss_two = F.cross_entropy(z_two[:n] / self.sinkhorn_kwargs.tau, target_two[:n])
        prediction_loss = 1/2 * (loss_one + loss_two) 
            
        alignment_loss = self.vicReg(h_one_batch, h_two_batch) # align embedding spaces 
        loss = alignment_loss + prediction_loss
        
        self.log_dict({
                f"{stage}_prediction_loss": prediction_loss, 
                f"{stage}_alignment_loss": alignment_loss, 
                f"{stage}_loss": loss
            })
        
        if batch_idx == 0 and stage == "val" and self.current_epoch % 5 == 0:
            if self.global_rank == 0: 
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f: 
                    fname = f.name 
                    self.plot_(
                        z_one[:n].detach(),
                        target_one[:n].detach(),
                        task=batch["task"], 
                        robot=batch["robot"],
                        fname=fname
                        )
                    
                    if isinstance(self.logger, pl.loggers.WandbLogger): 
                        self.logger.experiment.log({"tsne_plot": wandb.Image(fname)})
                     
                    os.remove(fname)
        
        return loss 
    
    def plot_(
        self, 
        x: TensorType["n", "k"],
        label: TensorType["n"], 
        task: TensorType["n"], 
        robot: TensorType["n"], 
        fname: str
        ) -> None:
        
        columns = ["x", "y", "label", "task", "robot"]
        
        x = x.cpu().numpy() # [n, k]
        label = label.argmax(-1).cpu().numpy() # [n, ]
        task = task.cpu().numpy().reshape(-1, ) # [n, ]
        robot = robot.cpu().numpy().reshape(-1, ) # [n, ]
        
        x = self.tsne.fit_transform(x) # [n, 2]
        data = np.stack(arrays=[x[:, 0], x[:, 1], label, task, robot], axis=-1)
        df = pd.DataFrame(data=data, columns=columns)
         
        plt.figure(figsize=(8, 6))
        scatterplot = sns.scatterplot(data=df, x="x", y="y", hue="label")
        fig = scatterplot.get_figure() 
        fig.savefig(fname)
        plt.close()
         
    @torch.no_grad()
    def distributed_sinkhorn(self, out: TensorType["n", "k"]) -> TensorType["n", "k"]:
    # from https://github.com/real-stanford/xskill/blob/main/xskill/model/core.py
    
        if self.trainer.world_size > 1: 
            out_gathered = self.all_gather(out, sync_grads=False)
            out = out_gathered.view(-1, out.shape[-1])
     
        Q = torch.exp(out / self.sinkhorn_kwargs.epsilon).T # [K, B]
        K, B = Q.shape # number of prototypes, number of samples to assign

        # matrix has to sum to 1
        sum_Q = torch.sum(Q) + 1e-8
        Q /= sum_Q

        for _ in range(self.sinkhorn_kwargs.sinkhorn_iterations):
            # normalize each row: total weight per prototype must be 1/K
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True) + 1e-8 # [K, 1]
            Q /= sum_of_rows 
            Q /= K

            # normalize each column: total weight per sample must be 1/B
            Q /= torch.sum(Q, dim=0, keepdim=True) + 1e-8 # [1, B]
            Q /= B

        Q *= B  # the colomns must sum to 1 so that Q is an assignment
        Q = Q.T 
        
        if self.trainer.world_size > 1: 
            batch_size_per_rank = out.shape[0] // self.trainer.world_size
            start_idx = self.global_rank * batch_size_per_rank
            end_idx = start_idx + batch_size_per_rank
            Q = Q[start_idx:end_idx]
            
        return Q