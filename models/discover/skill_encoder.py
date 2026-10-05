import os
import math
import wandb
import tempfile
import numpy as np
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
from models.utils.utils import SinusoidalEmbedding


class SkillEncoder(pl.LightningModule): 
    def __init__(
        self, 
        d_model: int, 
        num_plot: int, 
        abs_pe: bool, 
        queue_start_epochs: int, 
        freeze_c_epochs: int, 
        time_contrastive: bool, 
        temp_time_contrastive: float, 
        gripper_dropout_p: float, 
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
        _log_every_n_steps: Optional[int]=10, 
        ) -> None: 
        
        super().__init__()
        self.save_hyperparameters() 
        
        self.d_model = d_model
        self.num_plot = num_plot
        self.abs_pe = abs_pe
        self.queue_start_epochs = queue_start_epochs
        self.freeze_c_epochs = freeze_c_epochs
        self.time_contrastive = time_contrastive
        self.temp_time_contrastive = temp_time_contrastive
        self.gripper_dropout_p = gripper_dropout_p 
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
        self._log_every_n_steps = _log_every_n_steps
        
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
                nn.Linear(self.d_model, self.d_model * 2), 
                nn.ReLU(), 
                nn.Linear(self.d_model * 2, self.d_model),
            )
        self.sinusoidal_embedding = SinusoidalEmbedding(d_new=self.d_model)
        
        self.sequential = TransformerEncoder(**self.sequential_kwargs)
        
        self.C = nn.Linear(**self.prototype_kwargs.model) # [d_model, k]: config: bias=False 
        nn.init.xavier_uniform_(self.C.weight)
        
        with torch.no_grad():
            self.C.weight.copy_(F.normalize(self.C.weight, dim=1)) # normalize rows!
            
        self.down_emb = nn.Linear(self.d_model*3, self.d_model)    

        if self.with_uncertainty_weighting:
            self.uncertainty_weighting = UncertaintyWeighting(**self.uncertainty_weighting_kwargs)
        
        if self.time_contrastive: 
            self.timeContrastiveLoss = TimeContrastiveLoss(self.temp_time_contrastive)
        
        k = self.prototype_kwargs.model.out_features
        self.register_buffer("prototype_usage_ema", torch.full((k,), 1.0 / k)) # [k]    

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
    
    def train(self, mode: bool = True):
        super().train(mode)
        if self.vision_encoder_ckpt is not None:
            self.visionEncoder.eval()
        return self
    
    def on_train_epoch_start(self):
        if self.current_epoch < self.freeze_c_epochs:  
            self._set_freeze(True)
        
        elif self.current_epoch == self.freeze_c_epochs: 
            self._set_freeze(False)
    
    def _set_freeze(self, freeze: bool) -> None: 
        for p in self.C.parameters(): 
            p.requires_grad_(not freeze)
            
    def on_train_batch_end(self, outputs: Any, batch: Any, batch_idx: int) -> None:
        with torch.no_grad():
            self.C.weight.copy_(F.normalize(self.C.weight, dim=1))
            
    def on_validation_epoch_start(self):
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
            
            n_idx = min(self.num_plot, c_all.shape[0])
            n_plot = torch.randperm(c_all.shape[0], device="cpu")[:n_idx]
            
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f: 
                fname = f.name 
                self.tsne.plot_(
                    x=c_all[n_plot], 
                    label=target_all[n_plot], 
                    idxs=idxs_all[n_plot], 
                    task=task_all[n_plot], 
                    robot=robot_all[n_plot], 
                    fname=fname
                    )
                
            if isinstance(self.logger, pl.loggers.WandbLogger):
                self.logger.experiment.log({"tsne_plot": wandb.Image(fname)})
                    
            os.remove(fname)
    
    def on_test_start(self) -> None: 
        self.softDTW = SoftDTW(use_cuda=True, normalize=True)
    
    def predict_step(self, batch: Dict[str, Any], batch_idx: int, dataloader_idx:int=0) -> TensorType["batch_size", "k"]:
        with torch.no_grad(): 
            z = self(batch["rgb_one"], batch["rgb_two"], batch["g_qpos"], batch["conditions_idxs"]) 
            c = self.C(z) # [batch_size, k]: prototypes

            return c
        
    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:  
        return self._shared_step(batch, stage="train")
        
    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:  
        return self._shared_step(batch, stage="val")
        
    def _shared_step(self, batch: Dict[str, Any], stage: str) -> torch.Tensor: 
        losses = []
        batch_size, chunk = batch["rgb_one"].shape[:2]
        n = batch_size * chunk
        
        idxs = None
        if self.abs_pe: 
            idxs = batch["idxs"]

        z_anc = self(batch["rgb_one"], batch["rgb_two"], batch["g_qpos"], idxs=idxs) # anchor
        z_pos = self(batch["rgb_one_pos"], batch["rgb_two_pos"], batch["g_qpos_pos"], self.gripper_dropout_p, idxs=idxs) # positive sample 
        
        z_anc_full, z_pos_full = z_anc, z_pos
        
        with torch.no_grad():        
            if self.current_epoch >= self.queue_start_epochs and self.queue is not None:
                if self.queue.is_full and stage == "train": # only use queue once it is completely filled
                    queue_features = self.queue.get() # get all features 
                    z_anc_full = torch.cat([z_anc, queue_features[0]], dim=0) # [n+capacity, d_model]
                    z_pos_full = torch.cat([z_pos, queue_features[1]], dim=0) # [n+capacity, d_model]
                    
            if self.queue is not None and stage == "train":     
                self.queue.enqueue(torch.stack([z_anc.detach(), z_pos.detach()], dim=0))
                
        c_one = self.C(z_anc_full) # [n || n + capacity, k] 
        c_two = self.C(z_pos_full) # [n || n + capacity, k]
        
        c = {"one": c_one, "two": c_two}

        with torch.no_grad():
            q = {k: self.sinkhorn(v)[:n] for k, v in c.items()} # [n || n + capacity, k] each: targets
            
        p = {k: F.softmax(v[:n] / self.prototype_kwargs.tau, dim=-1) for k, v in c.items()} # [n, k]: predictions 
        log_p = {k: F.log_softmax(v[:n] / self.prototype_kwargs.tau, dim=-1) for k, v in c.items()} # [n, k]: predictions 
              
        if self.logger is not None: 
                with torch.no_grad(): 
                    _n, k = q["one"].shape
                    protoype_assignments = q["one"].argmax(dim=-1) # [n]
                    prototype_counts = torch.bincount(protoype_assignments, minlength=k).float() 
                    prototype_fraction = prototype_counts / _n # [k]
                    
                    if stage == "train":
                        self.prototype_usage_ema.mul_(0.99).add_(0.01 * prototype_fraction)                    
                    num_dead_prototypes = torch.sum(self.prototype_usage_ema < 0.1 / k) # []
                    
                    top_prototype_fraction = torch.max(prototype_fraction)
                    
                    prototype_mean = torch.mean(q["one"].float(), dim=0) # [k]
                    prototype_entropy = -torch.sum((prototype_mean * torch.log(prototype_mean+1e-8))) # []
                    prototype_perplexity = torch.exp(prototype_entropy) / k # []
                    
                    prediction_mean = torch.mean(p["one"].float(), dim=0) # [k]
                    prediction_entropy = -torch.sum((prediction_mean * torch.log(prediction_mean+1e-8))) # []
                    prediction_perplexity = torch.exp(prediction_entropy) / k # []
                    
                    per_sample_entropy = torch.mean(-torch.sum(p["one"] * log_p["one"], dim=-1) / math.log(k)) # normalized per sample entropy                      
                            
                    self.log_dict({
                        f"{stage}/num_dead_prototypes": num_dead_prototypes, 
                        f"{stage}/top_prototype_fraction": top_prototype_fraction, 
                        f"{stage}/prototype_perplexity": prototype_perplexity, 
                        f"{stage}/prediction_perplexity": prediction_perplexity, 
                        f"{stage}/per_sample_entropy": per_sample_entropy, 
                        
                    },
                    logger=True, 
                    prog_bar=False,
                    on_step=(stage == "train"),
                    on_epoch=True,
                    sync_dist=True
                    )
                    
                    if stage == "train" and self.global_step % self._log_every_n_steps == 0:
                        prototype_histogram = wandb.Histogram(np_histogram=(prototype_fraction.cpu().numpy(), np.arange(k+1)))
                        self.logger.experiment.log({
                            "train/prototype_histogram": prototype_histogram
                        })

        if self.time_contrastive: 
            t_one = c["one"][:n].view(batch_size, chunk, -1) # [batch_size, chunk, k]
            tcn_loss = self.timeContrastiveLoss(t_one) # []
            losses.extend([tcn_loss])
            
            self.log_dict({
                f"{stage}/time_contrastive_loss": tcn_loss.detach()
                },
                logger=True, 
                prog_bar=True,
                on_step=(stage == "train"),
                on_epoch=True,
                sync_dist=True
            )

        swav_losses = {}
        for target in c.keys():
            predictions = [k for k in c.keys() if k != target]
            swav_losses[target] = torch.mean(torch.stack([torch.mean(-torch.sum((q[prediction] * log_p[target]), dim=-1)) for prediction in predictions]))

        losses.extend(swav_losses.values())
    
        if stage == "val":             
            if sum(x.shape[0] for x in self._c_val) < self.num_plot: 
                self._c_val.append(c_one[:n].detach().cpu()) # [n, k]
                self._target_val.append(q["one"][:n].detach().cpu()) # [n, k]
                self._idxs_val.append(batch["idxs"][:, :, 0].reshape(n).cpu()) # [n]
                self._task_val.append(batch["task"].reshape(n).cpu()) # [n]
                self._robot_val.append(batch["robot"].reshape(n).cpu()) # [n]
        
        if self.with_uncertainty_weighting: 
            loss = self.uncertainty_weighting(losses) # []
        else: 
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
            sync_dist=True
        )
        
        return loss
    
    def forward(
        self, 
        rgb_one: TensorType["batch", "chunk", "window", "channels", "height", "width"], 
        rgb_two: TensorType["batch", "chunk", "window", "channels", "height", "width"], 
        g_qpos: TensorType["batch", "chunk", "window", "1"], 
        gripper_dropout_p: float=0.0, 
        idxs: Optional[TensorType["batch", "chunk", "window"]]=None
        ) -> torch.Tensor:
        
        batch_size, chunk, window = rgb_one.shape[:3]
        n = batch_size*chunk
                
        emb_one = self.visionEncoder(rgb_one) # [batch_size*chunk*window, d_model]
        emb_two = self.visionEncoder(rgb_two) # [batch_size*chunk*window, d_model]
        emb_gripper = self.gripperEncoder(self.sinusoidal_embedding(g_qpos.view(-1, 1))) # [batch_size*chunk*window, d_model]
        
        if self.training and gripper_dropout_p > 0.0:
            keep_p = 1.0 - gripper_dropout_p
            keep = torch.bernoulli(torch.full((n, 1, 1), keep_p, device=emb_gripper.device, dtype=emb_gripper.dtype))
            emb_gripper = (emb_gripper.view(n, window, -1) * keep / keep_p).view(n * window, -1)
        
        emb = torch.cat(tensors=(emb_one, emb_two, emb_gripper), dim=-1) # [batch_size*chunk*window, d_model*3]
        emb = self.down_emb(emb)
        emb = self.sequential(emb.view(n, window, -1), idxs=idxs) # [n, d_model]
        
        z = F.normalize(emb, dim=-1) # [n, d_model]
        
        return z