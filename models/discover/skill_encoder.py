import os 
import wandb
import tempfile
import numpy as np 
import pandas as pd 
import seaborn as sns
from typing import Dict, Union
from omegaconf import DictConfig
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt

from termcolor import colored

import torch
import torch.nn as nn 
import torch.nn.functional as F

import lightning.pytorch as pl 
from torchtyping import TensorType

from models.discover.utils.queue import FIFOQueue
from models.utils.loss import UncertaintyWeighting
from models.utils.vicreg import VICReg
from models.utils.vision import Transformer
from models.fine_tune.fine_tuner import FineTuner


class SkillEncoder(pl.LightningModule): 
    def __init__(
        self, 
        fine_tuner_ckpt: Union[str, os.PathLike],  
        sinkhorn_kwargs: DictConfig, 
        optimizer_kwargs: DictConfig, 
        vision_sequential_kwargs: DictConfig, 
        gripper_sequential_kwargs: DictConfig, 
        prototype_kwargs: DictConfig,  
        queue_kwargs: DictConfig, 
        tsne_kwargs: DictConfig, 
        ) -> None: 
        
        super().__init__()
        self.save_hyperparameters() 
        
        self.fine_tuner_ckpt = fine_tuner_ckpt
        self.sinkhorn_kwargs = sinkhorn_kwargs        
        self.optimizer_kwargs = optimizer_kwargs
        
        self.fineTuner = FineTuner.load_from_checkpoint(fine_tuner_ckpt)
        self.visionEncoder = self.fineTuner.visionEncoder
        self.gripperEncoder = self.fineTuner.gripperEncoder 
        self.sinusoidalEmbedding = self.fineTuner.sinusoidalEmbedding

        for param in self.visionEncoder.parameters(): 
                param.requires_grad = False 
        
        for param in self.gripperEncoder.parameters(): 
            param.requires_grad = False 
        
        self.visionEncoder.eval()
        self.gripperEncoder.eval() 
        
        self.visionSequential = Transformer(**vision_sequential_kwargs) 
        self.gripperSequential = Transformer(**gripper_sequential_kwargs) 
        
        self.prototype_emb = nn.Linear(**prototype_kwargs) 
        nn.init.orthogonal_(self.prototype_emb.weight)
    
        self.queue = FIFOQueue(**queue_kwargs)
        self.tsne = TSNE(**tsne_kwargs)

        self.weighted_loss = UncertaintyWeighting(num_losses=queue_kwargs.num_modalities)
        
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
        
    def on_after_backward(self):
            for name, param in self.named_parameters():
                if param.requires_grad and param.grad is None:
                    print(colored(f"Unused parameter: {name}", "red"))

    def training_step(self, batch, batch_idx) -> torch.Tensor:  
        return self(batch=batch, batch_idx=batch_idx, stage="train")
        
    def validation_step(self, batch, batch_idx) -> torch.Tensor:  
        return self(batch=batch, batch_idx=batch_idx, stage="val")

    def test_step(self, batch, batch_idx) -> torch.Tensor:        
        return self(batch=batch, batch_idx=batch_idx, stage="test")
    
    def forward(
        self, 
        batch: Dict[str, TensorType["batch", "chunk", "window", "*"]],
        batch_idx: int, 
        stage: str
        ) -> torch.Tensor:
        batch_size, chunk, window = batch["rgb_one"].shape[:3]
        n = batch_size*chunk*window
        
        rgb_one_emb = self.visionEncoder(batch["rgb_one"]) # [n, d_model]
        rgb_two_emb = self.visionEncoder(batch["rgb_two"]) # [n, d_model]
         
        gripper_x = batch["g_qpos"].flatten().unsqueeze(-1) # [n, 1]
        gripper_emb = self.gripperEncoder(self.sinusoidalEmbedding(gripper_x)) # [n, d_model]
        
        h_one = self.visionSequential(rgb_one_emb.view(batch_size*chunk, window, -1)) # [n, d_model]: robot0_eye_in_hand_view
        h_two =  self.visionSequential(rgb_two_emb.view(batch_size*chunk, window, -1)) # [n, d_model]: agentview_image
        h_gripper = self.gripperSequential(gripper_emb.view(batch_size*chunk, window, -1)) # [n, d_model]: gripper states
        
        # Normalize features to lie on unit sphere 
        h_one_batch = F.normalize(h_one, dim=-1)
        h_two_batch = F.normalize(h_two, dim=-1)
        h_gripper_batch = F.normalize(h_gripper, dim=-1)

        if self.queue is not None and self.queue.is_full: 
                queue_features = self.queue.dequeue() 
                h_one_assign = torch.cat([h_one_batch, queue_features[0]], dim=0)
                h_two_assign = torch.cat([h_two_batch, queue_features[1]], dim=0)
                h_gripper_assign = torch.cat([h_gripper_batch, queue_features[2]], dim=0)
        else: 
            h_one_assign = h_one_batch
            h_two_assign = h_two_batch
            h_gripper_assign = h_gripper_batch
            
        if self.queue is not None:   
            self.queue.enqueue(torch.stack([h_one_batch, h_two_batch, h_gripper_batch], dim=0))
   
        # Map to prototypes
        z_one = self.prototype_emb(h_one_assign) # [n, k]
        z_two = self.prototype_emb(h_two_assign) # [n, k]
        z_gripper = self.prototype_emb(h_gripper_assign) # [n, k]
        
        # Find pseudo-labels
        with torch.no_grad(): 
            target_one = self.distributed_sinkhorn(z_gripper) # [n, k]
            target_two = self.distributed_sinkhorn(z_one) # [n, k]
            target_three = self.distributed_sinkhorn(z_two) # [n, k]
        
        loss_one = F.cross_entropy(z_gripper[:n] / self.sinkhorn_kwargs.tau, target_one[:n])
        loss_two = F.cross_entropy(z_one[:n] / self.sinkhorn_kwargs.tau, target_two[:n])
        loss_gripper = F.cross_entropy(z_two[:n] / self.sinkhorn_kwargs.tau, target_three[:n])
        loss = 1/3 * (loss_one + loss_two + loss_gripper)
        
        self.log_dict(
            {
                f"{stage}_loss": loss
            },
                sync_dist=True
            )
        
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
        scatterplot = sns.scatterplot(data=df, x="x", y="y", hue="label", style="task", size="robot")
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