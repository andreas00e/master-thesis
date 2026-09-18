import os
import wandb
import tempfile
import numpy as np 
import pandas as pd 
import seaborn as sns
from hydra.utils import instantiate
from typing import Any, Dict, Optional
from omegaconf import DictConfig
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt

import torch
import torch.nn as nn 
import torch.nn.functional as F
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

import lightning.pytorch as pl 
from torchtyping import TensorType

from models.discover.utils.queue import FIFOQueue
from models.utils.loss import UncertaintyWeighting
from models.utils.aux_models import TransformerEncoder, VisionEncoder, CNN
from models.fine_tune.fine_tuner import FineTunerVisual, FineTunerGripper

TASK_DICT = {
    0: "square",
    1: "threading"
}

ROBOT_DICT = {
    0: "iiwa", 
    1: "panda",
    2: "sawyer", 
    3: "ur5e"
}

class SkillEncoder(pl.LightningModule): 
    def __init__(
        self, 
        d_model: int, 
        n_plot: int, 
        vision_encoder_ckpt: Optional[str],  
        gripper_encoder_ckpt: Optional[str],
        sequential_kwargs: DictConfig, 
        prototype_kwargs: DictConfig,  
        optimizer_kwargs: DictConfig, 
        lr_scheduler_kwargs: DictConfig, 
        uncertainty_weighting_kwargs: DictConfig,
        sinkhorn_kwargs: DictConfig, 
        queue_kwargs: DictConfig, 
        tsne_kwargs: DictConfig, 
        ) -> None: 
        
        super().__init__()
        self.save_hyperparameters() 
        
        self.d_model = d_model
        self.n_plot = n_plot
        self.vision_encoder_ckpt = vision_encoder_ckpt
        self.gripper_encoder_ckpt = gripper_encoder_ckpt
        self.sequential_kwargs = sequential_kwargs
        self.prototype_kwargs = prototype_kwargs
        self.optimizer_kwargs = optimizer_kwargs
        self.lr_scheduler_kwargs = lr_scheduler_kwargs
        self.uncertainty_weighting_kwargs = uncertainty_weighting_kwargs
        self.sinkhorn_kwargs = sinkhorn_kwargs        
        self.queue_kwargs = queue_kwargs
        self.tsne_kwargs = tsne_kwargs
        
        if vision_encoder_ckpt is not None: 
            self.visionEncoder = FineTunerVisual.load_from_checkpoint(vision_encoder_ckpt)
            self.visionEncoder.eval()
                
            for p in self.visionEncoder.parameters(): 
                p.requires_grad_(False)
        else: 
            self.visionEncoder = CNN(self.d_model)
        
        if gripper_encoder_ckpt is not None: 
            self.gripperEncoder = None # TODO: Exchange with fine-tuned gripper encoder 
        else: 
            self.gripperEncoder = nn.Sequential(
                nn.Linear(1, self.d_model // 2), 
                nn.ReLU(), 
                nn.Linear(self.d_model // 2, self.d_model)
            )  
        
        self.sequential = TransformerEncoder(**self.sequential_kwargs)
        
        self.C = nn.Linear(**self.prototype_kwargs) 
        nn.init.xavier_uniform_(self.C.weight)
        with torch.no_grad():
            self.C.weight.copy_(F.normalize(self.C.weight, dim=0))
        
        self.uncertainty_weighting = UncertaintyWeighting(**self.uncertainty_weighting_kwargs)
        self.queue = FIFOQueue(**self.queue_kwargs)
        self.tsne = TSNE(**self.tsne_kwargs)
        
        self._c_val = []
        self._target_val = []
        self._task_val = []
        self._robot_val = []
        
    def configure_optimizers(self) -> Dict[str, Any]:
        trainable_parameters = filter(lambda p: p.requires_grad, self.parameters())        
        optimizer = instantiate(self.optimizer_kwargs, params=trainable_parameters)
        
        if hasattr(self, "trainer") and self.trainer is not None: 
            total_steps = self.trainer.estimated_stepping_batches 
            
            warmup_percentage = self.lr_scheduler_kwargs.get("warmup_percentage", 0.1)
            warmup_steps = int(warmup_percentage * total_steps)
            decay_steps = total_steps - warmup_steps
            
            self.lr_scheduler_kwargs.linear.total_iters = warmup_steps 
            self.lr_scheduler_kwargs.cosine_annealing.T_max = decay_steps 

        scheduler_one = LinearLR(optimizer, **self.lr_scheduler_kwargs.linear)
        scheduler_two = CosineAnnealingLR(optimizer, **self.lr_scheduler_kwargs.cosine_annealing)

        scheduler = SequentialLR(optimizer, schedulers=[scheduler_one, scheduler_two],  milestones=[warmup_steps])
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler, 
                "interval": "step", 
                "frequency": 1
            }
        }

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:  
        return self(batch, batch_idx, stage="train")
        
    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:  
        return self(batch, batch_idx, stage="val")

    def test_step(self, batch: Any, batch_idx: int) -> torch.Tensor:        
        return self(batch, batch_idx, stage="test")

    def on_train_batch_end(self, outputs: Any, batch: Any, batch_idx: int) -> None:
        with torch.no_grad():
            self.C.weight.copy_(F.normalize(self.C.weight, dim=0))
    
    def on_validation_epoch_start(self) -> None:   
        self._c_val.clear()
        self._target_val.clear()
        self._task_val.clear()
        self._robot_val.clear()
    
    def on_validation_epoch_end(self) -> None:
        if self.global_rank == 0 and len(self._c_val) != 0:
            c_all = torch.cat(self._c_val)
            target_all = torch.cat(self._target_val)
            task_all = torch.cat(self._task_val)
            robot_all = torch.cat(self._robot_val)
            
            n_plot = min(self.n_plot, c_all.shape[0])
            
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f: 
                fname = f.name 
                self.plot_(
                    x=c_all[:n_plot], 
                    label=target_all[:n_plot], 
                    task=task_all[:n_plot], 
                    robot=robot_all[:n_plot], 
                    fname=fname
                    )
                
                if isinstance(self.logger, pl.loggers.WandbLogger): 
                    self.logger.experiment.log({"tsne_plot": wandb.Image(fname)})
                    
            os.remove(fname)

    def forward(
        self, 
        batch: Dict[str, TensorType["batch", "chunk", "window", "*"]],
        batch_idx: int, 
        stage: str
        ) -> torch.Tensor:
        
        with torch.no_grad(): 
            self.C.weight.copy_(F.normalize(self.C.weight, dim=0))
    
        batch_size, chunk, window = batch["rgb_one"].shape[:3]
        n = batch_size*chunk
        
        emb_one = self.visionEncoder(batch["rgb_one"]) # [batch_size*chunk*window, d_model]
        emb_two = self.visionEncoder(batch["rgb_one_pos"])
        # emb_two = self.visionEncoder(batch["rgb_two"]) # [batch_size*chunk*window, d_model]
         
        g_qpos = batch["g_qpos"] # [batch, chunk, window, 1]
        g_qpos = g_qpos.view(-1, 1) # [batch_size*chunk*window, 1]
        emb_gripper = self.gripperEncoder(g_qpos) # [batch_size*chunk*window, d_model]
        
        # Non-linear mapping 
        emb_one = self.sequential(emb_one.view(batch_size*chunk, window, -1)) # [n, d_model]: robot0_eye_in_hand_view
        emb_two =  self.sequential(emb_two.view(batch_size*chunk, window, -1)) # [n, d_model]: agentview_image
        emb_gripper = self.sequential(emb_gripper.view(batch_size*chunk, window, -1)) # [n, d_model]: gripper states
        
        # Project features to the unit sphere
        z_one = F.normalize(emb_one, dim=-1) # [n, d_model]
        z_two = F.normalize(emb_two, dim=-1) # [n, d_model]
        z_gripper = F.normalize(emb_gripper, dim=-1) # [n, d_model]
        
        if self.trainer and self.trainer.training: 
            if self.queue is not None:
                if self.queue.is_full:
                    queue_features = self.queue.get() # all features 
                    z_one = torch.cat([z_one, queue_features[0]], dim=0)
                    z_two = torch.cat([z_two, queue_features[1]], dim=0)
                    z_gripper = torch.cat([z_gripper, queue_features[2]], dim=0)
                    
                else: 
                    self.queue.enqueue(torch.stack([z_one.detach(), z_two.detach(), z_gripper.detach()], dim=0))
   
        # Map to prototypes
        c_one = self.C(z_one) # [n, k]
        c_two = self.C(z_two) # [n, k]
        c_gripper = self.C(z_gripper) # [n, k]
        
        # Find pseudo-labels
        with torch.no_grad(): 
            q_one = self.distributed_sinkhorn(c_gripper) # [n, k]
            q_two = self.distributed_sinkhorn(c_one) # [n, k]
            q_gripper = self.distributed_sinkhorn(c_two) # [n, k]
        
        p_one = F.log_softmax(c_one[:n, :] / self.sinkhorn_kwargs.tau, dim=-1)
        p_two = F.log_softmax(c_two[:n, :] / self.sinkhorn_kwargs.tau, dim=-1)
        p_gripper = F.log_softmax(c_gripper[:n, :] / self.sinkhorn_kwargs.tau, dim=-1)
    
        if self.trainer is not None and self.trainer.validating: 
            self._c_val.append(c_one[:n, :].detach().cpu())
            self._target_val.append(q_one.detach().cpu())
            self._task_val.append(batch["task"].cpu())
            self._robot_val.append(batch["robot"].cpu())
        
        loss_one = -torch.mean(torch.sum(q_one[:n] * p_one, dim=-1))
        loss_two = -torch.mean(torch.sum(q_two[:n] * p_two, dim=-1))
        loss_gripper = -torch.mean(torch.sum(q_gripper[:n] * p_gripper, dim=-1))
        
        loss = self.uncertainty_weighting([loss_one, loss_two, loss_gripper]) # []
    
        self.log_dict(
            {
                f"{stage}/loss_one": loss_one, 
                f"{stage}/loss_two": loss_two, 
                f"{stage}/loss_gripper": loss_gripper, 
                f"{stage}/loss": loss
             
            },
            logger=True,
            prog_bar=True, 
            on_step=stage=="train", 
            on_epoch=True, 
            sync_dist=True, 
        )
        
        return loss
    
    def plot_(
        self, 
        x: TensorType["n", "k"],
        label: TensorType["n"], 
        task: TensorType["n"], 
        robot: TensorType["n"], 
        fname: str
        ) -> None:
                
        x = x.cpu().numpy() # [n, k]
        label = label.argmax(-1).cpu().numpy() # [n]
        task = task.cpu().numpy().reshape(-1) # [n]
        robot = robot.cpu().numpy().reshape(-1) # [n]
        
        x = self.tsne.fit_transform(x) # [n, 2]
        
        df = pd.DataFrame({
            "x": x[:, 0], 
            "y": x[:, 1], 
            "label": label, 
            })
        
        df["task"] = task.astype(int).map(TASK_DICT)
        df["robot"] = robot.astype(int).map(ROBOT_DICT)
         
        plt.figure(figsize=(8, 6))
        scatterplot = sns.scatterplot(
            data=df, 
            x="x",
            y="y", 
            hue="task", 
            style="robot"
            )
        
        fig = scatterplot.get_figure() 
        fig.savefig(fname)
        plt.close()
         
    @torch.no_grad()
    def distributed_sinkhorn(self, out: TensorType["n", "k"]) -> TensorType["n", "k"]:
    # from https://github.com/real-stanford/xskill/blob/main/xskill/model/core.py
    
        if self.trainer.world_size > 1: 
            out_gathered = self.all_gather(out, sync_grads=False)
            out = out_gathered.reshape(-1, out.shape[-1])
     
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