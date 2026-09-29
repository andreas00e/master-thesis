import os
import math
import wandb
import tempfile
from hydra.utils import instantiate
from typing import Any, Dict, Optional
from omegaconf import DictConfig

import torch
import torch.nn as nn 
import torch.nn.functional as F
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

import lightning.pytorch as pl 
from torchtyping import TensorType

from models.fine_tune.fine_tuner import FineTunerVisual
from models.discover.utils.queue import FIFOQueue
from models.discover.utils.sinkhorn import Sinkhorn
from models.discover.utils.visualize import Visualize
from models.utils.aux_models import TransformerEncoder, VisionEncoder, CNN
from models.utils.loss import UncertaintyWeighting, TimeContrastiveLoss
from models.utils.soft_dtw_cuda import SoftDTW


class SkillEncoder(pl.LightningModule): 
    def __init__(
        self, 
        d_model: int, 
        num_plot: int, 
        queue_start_epochs: int, 
        with_both_viewpoints: bool, 
        time_contrastive: bool, 
        temp_time_contrastive: float, 
        with_uncertainty_weighting: bool, 
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
        self.with_both_viewpoints = with_both_viewpoints
        self.time_contrastive = time_contrastive
        self.temp_time_contrastive = temp_time_contrastive 
        self.with_uncertainty_weighting = with_uncertainty_weighting
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
            for p in self.visionEncoder.parameters(): 
                p.requires_grad = False
        else: 
            self.visionEncoder = CNN(self.d_model)
        
        if gripper_encoder_ckpt is not None: 
            raise NotImplementedError
        else: 
            self.gripperEncoder = nn.Sequential(
                nn.Linear(1, self.d_model // 2), 
                nn.ReLU(), 
                nn.Linear(self.d_model // 2, self.d_model)
            )  
        
        self.sequential = TransformerEncoder(**self.sequential_kwargs)
        
        self.C = nn.Linear(**self.prototype_kwargs.model) # [d_model, k]: config: bias=False 
        nn.init.xavier_uniform_(self.C.weight)
        
        with torch.no_grad():
            self.C.weight.copy_(F.normalize(self.C.weight, dim=1)) # normalize rows!
            
        if self.with_both_viewpoints: 
            self.down_emb = nn.Linear(self.d_model*2, d_model)    
        
        if self.with_uncertainty_weighting:
            self.uncertainty_weighting = UncertaintyWeighting(**self.uncertainty_weighting_kwargs)
        
        if self.time_contrastive: 
            self.timeContrastiveLoss = TimeContrastiveLoss(self.temp_time_contrastive)
        
        self.softDTW = None
        
        self.queue = FIFOQueue(**self.queue_kwargs)
        self.sinkhorn = Sinkhorn(**self.sinkhorn_kwargs)
        self.tsne = Visualize(self.tsne_kwargs)
    
        self._c_val = []
        self._target_val = []
        self._idxs_val = []
        self._task_val = []
        self._robot_val = []
                
    def configure_optimizers(self) -> Dict[str, Any]:
        trainable_parameters = filter(lambda p: p.requires_grad, self.parameters())        
        optimizer = instantiate(self.optimizer_kwargs, params=trainable_parameters)
        
        warmup_steps = 100
        if hasattr(self, "trainer") and self.trainer is not None and math.isfinite(self.trainer.estimated_stepping_batches):   
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
    
    def predict_step(self, batch: Any):
        batch_size, chunk, window = batch["rgb_one"].shape[:3]
        n = batch_size*chunk
        
        # 1. Feature Extraction 
        emb_one = self.visionEncoder(batch["rgb_one"]) # [batch_size*chunk*window, d_model]
        
        if self.with_both_viewpoints: 
            emb_one_pos = self.visionEncoder(batch["rgb_one_pos"]) # [batch_size*chunk*window, d_model]            
            emb_one = torch.cat(tensors=(emb_one, emb_one_pos), dim=-1) # [batch_size*chunk*window, d_model*2]            
            emb_one = self.down_emb(emb_one)

        # Sequential Transformer Encoding 
        emb_one = self.sequential(emb_one.view(n, window, -1)) # [n, d_model]: robot0_eye_in_hand_view
        
        # Unit Sphere Normalization 
        z_one = F.normalize(emb_one, dim=-1) # [n, d_model]
        
        c_one = self.C(z_one) # [n, k] 

        return c_one
    
    def train(self, mode: bool = True):
        super().train(mode)
        if self.vision_encoder_ckpt is not None:
            self.visionEncoder.eval()
        return self
            
    def on_train_batch_end(self, outputs: Any, batch: Any, batch_idx: int) -> None:
        with torch.no_grad():
            self.C.weight.copy_(F.normalize(self.C.weight, dim=1))
    
    def on_validation_epoch_start(self) -> None:   
        self._c_val.clear()
        self._target_val.clear()
        self._idxs_val.clear()
        self._task_val.clear()
        self._robot_val.clear()
    
    def on_validation_epoch_end(self) -> None:
        if len(self._c_val) != 0:
            c_all = torch.cat(self._c_val)
            target_all = torch.cat(self._target_val)
            idxs_all = torch.cat(self._idxs_val)
            task_all = torch.cat(self._task_val)
            robot_all = torch.cat(self._robot_val)
            
            n_plot = min(self.num_plot, c_all.shape[0])
            
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f: 
                fname = f.name 
                self.tsne.plot_(
                    x=c_all[:n_plot], 
                    label=target_all[:n_plot], 
                    idxs=idxs_all[:n_plot], 
                    task=task_all[:n_plot], 
                    robot=robot_all[:n_plot], 
                    fname=fname
                    )
                
                if isinstance(self.logger, pl.loggers.WandbLogger): 
                    self.logger.experiment.log({"tsne_plot": wandb.Image(fname)})
                    
            os.remove(fname)
    
    def on_test_start(self) -> None: 
        self.softDTW = SoftDTW(use_cuda=True, normalize=True)
    
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
        emb_two = self.visionEncoder(batch["rgb_two"])
        emb_gripper = self.gripperEncoder(batch["g_qpos"].view(-1, 1))
        
        if self.with_both_viewpoints: 
            emb_one_pos = self.visionEncoder(batch["rgb_one_pos"]) # [batch_size*chunk*window, d_model]
            emb_two_pos = self.visionEncoder(batch["rgb_two_pos"]) # [batch_size*chunk*window, d_model]
            
            emb_one = torch.cat(tensors=(emb_one, emb_one_pos), dim=-1) # [batch_size*chunk*window, d_model*2]
            emb_two = torch.cat(tensors=(emb_two, emb_two_pos), dim=-1) # [batch_size*chunk*window, d_model*2]
            
            emb_one = self.down_emb(emb_one)
            emb_two = self.down_emb(emb_two)

        # Sequential Transformer Encoding 
        emb_one = self.sequential(emb_one.view(n, window, -1)) # [n, d_model]: robot0_eye_in_hand_view
        emb_two =  self.sequential(emb_two.view(n, window, -1)) # [n, d_model]: agentview_image
        emb_gripper = self.sequential(emb_gripper.view(n, window, -1)) # [n, d_model]: gripper states
        
        # Unit Sphere Normalization 
        z_one = F.normalize(emb_one, dim=-1) # [n, d_model]
        z_two = F.normalize(emb_two, dim=-1) # [n, d_model]
        z_gripper = F.normalize(emb_gripper, dim=-1) # [n, d_model]
        
        z_one_full, z_two_full, z_gripper_full = z_one, z_two, z_gripper
                
        if self.current_epoch >= self.queue_start_epochs and self.queue is not None:
            if self.queue.is_full and stage == "train": # only use queue once it is completely filled
                queue_features = self.queue.get() # get all features 
                z_one_full = torch.cat([z_one, queue_features[0]], dim=0) # [n+capacity, d_model]
                z_two_full = torch.cat([z_two, queue_features[1]], dim=0) # [n+capacity, d_model]
                z_gripper_full = torch.cat([z_gripper, queue_features[2]], dim=0)
                
        if self.queue is not None and stage == "train":     
            self.queue.enqueue(torch.stack([z_one.detach(), z_two.detach(), z_gripper.detach()], dim=0))
                
        # 3. Prototype Mapping
        c_one = self.C(z_one_full) # [n || n + capacity, k] 
        c_two = self.C(z_two_full) # [n || n + capacity, k]
        c_gripper = self.C(z_gripper_full) # [n || n + capacity, k]
        
        c = {"one": c_one, "two": c_two, "grip": c_gripper}
        
        # 4. Sinkhorn Assignment (Over current batch + queue)
        with torch.no_grad():
            q = {k: self.sinkhorn(s) for k, s in c.items()} # [n || n + capacity, k] each: targets
            
        # Softmax probabilities for current batch elements 
        p = {k: F.log_softmax(s[:n] / self.prototype_kwargs.tau, dim=-1) for k, s in c.items()} # [n, k]: predictions 

        # Time Contrastive Loss (TCN) - Only computed on current batch elements 
        if self.time_contrastive: 
            t_one = c["one"][:n].view(batch_size, chunk, -1) # [batch_size, chunk, k]
            tcn_loss = self.timeContrastiveLoss(t_one) # []
            losses.extend([tcn_loss])
            
            self.log_dict({
                f"{stage}/time_contrastive_loss": tcn_loss.detach()
            })

        # one loss per predicting stream: it predicts the codes of the other two
        swav_losses = {}
        for target in c.keys():
            predictions = [s for s in c.keys() if s != target]
            swav_losses[target] = torch.mean(torch.stack([torch.mean(-torch.sum((q[prediction] * p[target]), dim=-1)) for prediction in predictions]))

        losses.extend(swav_losses.values()) # 3 (+1 TCN) -> num_losses: 4
    
        if stage == "val": 
            if sum(x.shape[0] for x in self._c_val) < self.num_plot: 
                self._c_val.append(c_one[:n].detach().cpu()) # [n, k]
                self._target_val.append(q["one"][:n].detach().cpu()) # [n, k]
                self._idxs_val.append(batch["idxs"][:, :, 0].view(n).cpu()) # [n]
                self._task_val.append(batch["task"].view(n).cpu()) # [n]
                self._robot_val.append(batch["robot"].view(n).cpu()) # [n]
        
        if self.with_uncertainty_weighting: 
            pass
            # loss = self.uncertainty_weighting(losses) # []
        # else: 
        loss = torch.mean(torch.stack(losses)) # []

        self.log_dict(
            {
                **{f"{stage}/loss_{k}": v.detach() for k, v in swav_losses.items()},
                f"{stage}/loss": loss.detach(),
            },
            logger=True, 
            prog_bar=True,
            on_step=(stage == "train"), 
            on_epoch=True,
            sync_dist=True,
        )
        
        return loss