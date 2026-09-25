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
from models.utils.loss import UncertaintyWeighting, TimeContrastiveLoss
from models.utils.aux_models import TransformerEncoder, VisionEncoder, CNN
from models.fine_tune.fine_tuner import FineTunerVisual


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

map_task = np.vectorize(lambda x: TASK_DICT.get(x, str(x)))
map_robot =np.vectorize(lambda x: ROBOT_DICT.get(x, str(x)))


class SkillEncoder(pl.LightningModule): 
    def __init__(
        self, 
        d_model: int, 
        num_plot: int, 
        queue_start_epochs: int, 
        both_viewpoints: bool, 
        time_contrastive: bool, 
        temp_time_contrastive: float, 
        uncertainty_weighting: bool, 
        uncertainty_weighting_kwargs: DictConfig,
        vision_encoder_ckpt: Optional[str],  
        gripper_encoder_ckpt: Optional[str],
        sequential_kwargs: DictConfig, 
        prototype_kwargs: DictConfig,  
        optimizer_kwargs: DictConfig, 
        lr_scheduler_kwargs: DictConfig, 
        sinkhorn_kwargs: DictConfig, 
        queue_kwargs: DictConfig, 
        tsne_kwargs: DictConfig, 
        ) -> None: 
        
        super().__init__()
        self.save_hyperparameters() 
        
        self.d_model = d_model
        self.num_plot = num_plot
        self.queue_start_epochs = queue_start_epochs
        self.both_viewpoints = both_viewpoints
        self.time_contrastive = time_contrastive
        self.temp_time_contrastive = temp_time_contrastive 
        self.uncertainty_weighting = uncertainty_weighting
        self.uncertainty_weighting_kwargs = uncertainty_weighting_kwargs
        self.vision_encoder_ckpt = vision_encoder_ckpt
        self.gripper_encoder_ckpt = gripper_encoder_ckpt
        self.sequential_kwargs = sequential_kwargs
        self.prototype_kwargs = prototype_kwargs
        self.optimizer_kwargs = optimizer_kwargs
        self.lr_scheduler_kwargs = lr_scheduler_kwargs
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
        self.C = nn.Linear(**self.prototype_kwargs) # config: bias=False 
        nn.init.xavier_uniform_(self.C.weight)
        with torch.no_grad():
            self.C.weight.copy_(F.normalize(self.C.weight, dim=0))
        if self.both_viewpoints: 
            self.down_emb = nn.Linear(self.d_model*2, d_model)    
        
        if self.uncertainty_weighting:
            self.uncertainty_weighting = UncertaintyWeighting(**self.uncertainty_weighting_kwargs)
        self.timeContrastiveLoss = TimeContrastiveLoss(self.temp_time_contrastive)

        self.queue = FIFOQueue(**self.queue_kwargs)
        self.tsne = TSNE(**self.tsne_kwargs)
    
        self._c_val = []
        self._target_val = []
        self._task_val = []
        self._robot_val = []
                
    def configure_optimizers(self) -> Dict[str, Any]:
        trainable_parameters = filter(lambda p: p.requires_grad, self.parameters())        
        optimizer = instantiate(self.optimizer_kwargs, params=trainable_parameters)
        
        warmup_steps = 100
        if hasattr(self, "trainer") and self.trainer is not None and self.trainer.estimated_stepping_batches:            
            total_steps = self.trainer.estimated_stepping_batches 
            warmup_percentage = self.lr_scheduler_kwargs.get("warmup_percentage", 0.1)
            warmup_steps = int(warmup_percentage * total_steps)
            decay_steps = max(1, total_steps - warmup_steps)
            
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
        batch_size, chunk, window = batch["rgb_one"].shape[:3]
        n = batch_size*chunk
        
        emb_one = self.visionEncoder(batch["rgb_one"]) # [batch_size*chunk*window, d_model]
        emb_two = self.visionEncoder(batch["rgb_one_pos"])
            
        if self.both_viewpoints:         
            emb_three = self.visionEncoder(batch["rgb_two"])
            emb_four = self.visionEncoder(batch["rgb_two_pos"])
        
            emb = torch.cat((emb_one, emb_three), dim=-1) # [batch_size*chunk*window, d_model*2]
            emb_pos = torch.cat((emb_two, emb_four), dim=-1) # [batch_size*chunk*window, d_model*2]
            
            emb_one = self.down_emb(emb)
            emb_two = self.down_emb(emb_pos)
        
        emb_one = self.sequential(emb_one.view(n, window, -1)) # [n, d_model]: robot0_eye_in_hand_view
        emb_two =  self.sequential(emb_two.view(n, window, -1)) # [n, d_model]: agentview_image
        
        z_one = F.normalize(emb_one, dim=-1) # [n, d_model]
        z_two = F.normalize(emb_two, dim=-1) # [n, d_model]
        
        c_one = self.C(z_one_full) # [n, k] 
        c_two = self.C(z_two_full) # [n, k]

    def on_train_epoch_start(self) -> None:
        if hasattr(self, "queue") and self.queue is not None:
            self.queue.reset()

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
            
            n_plot = min(self.num_plot, c_all.shape[0])
            
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
        
        losses = []

        batch_size, chunk, window = batch["rgb_one"].shape[:3]
        n = batch_size*chunk
        
        # 1. Feature Extraction 
        emb_one = self.visionEncoder(batch["rgb_one"]) # [batch_size*chunk*window, d_model]
        emb_two = self.visionEncoder(batch["rgb_one_pos"])
        
        # if self.gripper:       
        #     g_qpos = batch["g_qpos"].view(-1, 1) # [batch_size*chunk*window, 1]
        #     emb_gripper = self.gripperEncoder(g_qpos) # [batch_size*chunk*window, d_model]
            
        if self.both_viewpoints:         
            emb_three = self.visionEncoder(batch["rgb_two"])
            emb_four = self.visionEncoder(batch["rgb_two_pos"])
        
            emb = torch.cat((emb_one, emb_three), dim=-1) # [batch_size*chunk*window, d_model*2]
            emb_pos = torch.cat((emb_two, emb_four), dim=-1) # [batch_size*chunk*window, d_model*2]
            
            emb_one = self.down_emb(emb)
            emb_two = self.down_emb(emb_pos)
        
        # Sequential Transformer Encoding 
        emb_one = self.sequential(emb_one.view(n, window, -1)) # [n, d_model]: robot0_eye_in_hand_view
        emb_two =  self.sequential(emb_two.view(n, window, -1)) # [n, d_model]: agentview_image
        # emb_gripper = self.sequential(emb_gripper.view(n, window, -1)) # [n, d_model]: gripper states
        
        # Unit Sphere Normalization 
        z_one = F.normalize(emb_one, dim=-1) # [n, d_model]
        z_two = F.normalize(emb_two, dim=-1) # [n, d_model]
        # z_gripper = F.normalize(emb_gripper, dim=-1) # [n, d_model]
        
        if self.curret_epoch >= self.queue_epoch_start:
            if stage == "train" and self.queue is not None and self.queue.is_full:
                queue_features = self.queue.get() # all features 
                z_one_full = torch.cat([z_one, queue_features[0]], dim=0)
                z_two_full = torch.cat([z_two, queue_features[1]], dim=0)
                #z_gripper_full = torch.cat([z_gripper, queue_features[2]], dim=0)
            else: 
                z_one_full, z_two_full = z_one, z_two
                # z_one_full, z_two_full, z_gripper_full = z_one, z_two, z_gripper
                        
            if stage == "train" and self.queue is not None:  
                self.queue.enqueue(torch.stack([z_one.detach(), z_two.detach()], dim=0))
                # self.queue.enqueue(torch.stack([z_one.detach(), z_two.detach(), z_gripper.detach()], dim=0))
        else: 
            z_one_full, z_two_full = z_one, z_two
        
        # 3. Prototype Mapping
        c_one = self.C(z_one_full) # [n, k] # batch_size, chunk 
        c_two = self.C(z_two_full) # [n, k]
        # c_gripper = self.C(z_gripper_full) # [n, k]
        
        # Time Contrastive Loss (TCN) - Only computed on current batch elements 
        if self.time_contrastive: # only for one clip 
            t_one = c_one[:n].view(batch_size, chunk, -1)
            # t_two = c_two[:n].view(batch_size, chunk, -1)
            # t_gripper = c_gripper[:n].view(batch_size, chunk, -1)
            
            tcn_loss_one = self.timeContrastiveLoss(t_one)
            # tcn_loss_two = self.timeContrastiveLoss(t_two)
            # tcn_loss_three = self.timeContrastiveLoss(t_gripper)
            
            losses.extend([tcn_loss_one])
            # losses.extend([tcn_loss_one, tcn_loss_two])
            # losses.extend([tcn_loss_one, tcn_loss_two, tcn_loss_three])
            
        # 4. Sinkhorn Assignment (Over current batch + queue)
        if stage == "train": 
            with torch.no_grad(): 
                q_one = self.distributed_sinkhorn(c_two) # [n, k]
                q_two = self.distributed_sinkhorn(c_one) # [n, k]
                # q_gripper = self.distributed_sinkhorn(c_two) # [n, k]
        else: 
            q_one = F.softmax(c_two / self.sinkhorn_kwargs.tau, dim=-1)
            q_two = F.softmax(c_one / self.sinkhorn_kwargs.tau, dim=-1)
        
        # Softmax probabilities for current batch elements 
        p_one = F.log_softmax(c_one[:n, :] / self.sinkhorn_kwargs.tau, dim=-1)
        p_two = F.log_softmax(c_two[:n, :] / self.sinkhorn_kwargs.tau, dim=-1)
        # p_gripper = F.log_softmax(c_gripper[:n, :] / self.sinkhorn_kwargs.tau, dim=-1)
    
        if stage == "val": 
            self._c_val.append(c_one[:n, :].detach().cpu())
            self._target_val.append(q_one[:n].detach().cpu())
            self._task_val.append(batch["task"].cpu())
            self._robot_val.append(batch["robot"].cpu())
        
        # Cross-entropy loss on current batch targets
        loss_one = -torch.mean(torch.sum(q_one[:n] * p_one, dim=-1))
        loss_two = -torch.mean(torch.sum(q_two[:n] * p_two, dim=-1))
        # loss_gripper = -torch.mean(torch.sum(q_gripper[:n] * p_gripper, dim=-1))
        
        losses.extend([loss_one, loss_two])
        # losses.extend([loss_one, loss_two, loss_gripper])
        
        if self.uncertainty_weighting: 
            loss = self.uncertainty_weighting(losses) # []
        else: 
            loss = torch.mean(torch.stack(losses))
    
        self.log_dict(
            {
                f"{stage}/loss_one": loss_one.detach(), 
                f"{stage}/loss_two": loss_two.detach(), 
                # f"{stage}/loss_gripper": loss_gripper.detach(), 
                f"{stage}/loss": loss.detach()
             
            },
            logger=True,
            prog_bar=True, 
            on_step=(stage=="train"), 
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
        
        df["task"] = map_task(task.astype(int))
        df["robot"] =  map_robot(robot.astype(int))
         
        plt.figure(figsize=(8, 6))
        scatterplot = sns.scatterplot(
            data=df, 
            x="x",
            y="y", 
            hue="robot", 
            style="task"
            )
        
        fig = scatterplot.get_figure() 
        fig.savefig(fname)
        plt.close()
         
    @torch.no_grad()
    def distributed_sinkhorn(self, out: TensorType["n", "k"]) -> TensorType["n", "k"]:
    # from https://github.com/real-stanford/xskill/blob/main/xskill/model/core.py
     
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
            
        return Q