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

from models.discover.utils.queue import FIFOQueue
from models.discover.utils.sinkhorn import Sinkhorn
from models.discover.utils.visualize import Visualize
from models.utils.cnns import DinoV2
from models.utils.aux_models import TransformerEncoder
from models.utils.loss import TimeContrastiveLoss
from models.utils.utils import SinusoidalEmbedding


class SkillEncoder(pl.LightningModule): 
    def __init__(
        self, 
        d_model: int, 
        num_plot: int, 
        ema_alpha: float, 
        modalities: list[str], 
        queue_start_epochs: int, 
        freeze_c_epochs: int, 
        time_contrastive: bool, 
        temp_time_contrastive: float, 
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
        self.ema_alpha = ema_alpha
        self.modalities = modalities
        self.num_modalities = len(self.modalities)
        self.queue_start_epochs = queue_start_epochs
        self.freeze_c_epochs = freeze_c_epochs
        self.time_contrastive = time_contrastive
        self.temp_time_contrastive = temp_time_contrastive
        self.sequential_kwargs = sequential_kwargs
        self.prototype_kwargs = prototype_kwargs
        self.optimizer_kwargs = optimizer_kwargs
        self.lr_scheduler_kwargs = lr_scheduler_kwargs
        self.sinkhorn_kwargs = sinkhorn_kwargs        
        self.queue_kwargs = queue_kwargs
        self.tsne_kwargs = tsne_kwargs
        self._log_every_n_steps = _log_every_n_steps
        
        self.visionEncoder = DinoV2()
        self.visionEncoder.eval() 
        for param in self.visionEncoder.parameters(): 
            param.requires_grad_(False)
        
        trainable_modules = []
        if "rgb_one" in self.modalities: 
            self.visionOneEncoder = nn.Sequential(
                nn.Linear(384, self.d_model), 
                nn.LayerNorm(self.d_model)
                )
            trainable_modules.append(self.lNVisionOne)
        if "rgb_two" in self.modalities: 
            self.lNVisionTwo = nn.Sequential(
                nn.Linear(384, self.d_model), 
                nn.LayerNorm(self.d_model)
                )
            trainable_modules.append(self.lNVisionTwo)
        if "g_qpos" in self.modalities: 
            self.sinusoidal_embedding = SinusoidalEmbedding(self.d_model)
            self.gripperEncoder = nn.Sequential(
                nn.Linear(self.d_model, self.d_model), 
                nn.ReLU(), 
                nn.Linear(self.d_model, self.d_model), 
                nn.LayerNorm(self.d_model)
            ) 
            trainable_modules.append(self.gripperEncoder)
        
        _features = self.d_model*self.num_modalities
        self.downEmb = nn.Sequential(
            nn.Linear(_features, _features), 
            nn.ReLU(),
            nn.Linear(_features, self.d_model)    
        )
        trainable_modules.append(self.downEmb)
        
        self.sequential = TransformerEncoder(**self.sequential_kwargs)
        self.C = nn.Linear(**self.prototype_kwargs.model) # [k, d_model]: config: bias=False 
        
        if self.time_contrastive: 
            self.timeContrastiveLoss = TimeContrastiveLoss(self.temp_time_contrastive)
        
        k = self.prototype_kwargs.model.out_features
        self.register_buffer("prototype_usage_ema", torch.full((k,), 1.0 / k)) # [k]  
                
        self.queue = FIFOQueue(**self.queue_kwargs)
        self.sinkhorn = Sinkhorn(**self.sinkhorn_kwargs)
        self.tsne = Visualize(self.tsne_kwargs)
    
        self._c_val = []
        self._target_val = []
        self._idxs_val = []
        self._task_val = []
        self._robot_val = []
        
        for m in trainable_modules:
             m.apply(self._init_weights)
        nn.init.xavier_uniform_(self.C.weight)

    def _init_weights(self, module): 
        if isinstance(module, nn.Linear): 
            nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
            if module.bias is not None: 
                nn.init.zeros_(module.bias)      
                
        if isinstance(module, nn.LayerNorm): 
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)         
                
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

    def on_train_epoch_start(self):
        self._set_freeze(self.current_epoch < self.freeze_c_epochs)
    
    def _set_freeze(self, freeze: bool) -> None: 
        for p in self.C.parameters(): 
            p.requires_grad_(not freeze)
            
    def on_train_batch_start(self, batch: Dict[str, Any], batch_idx: int) -> None:
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
    
    def predict_step(self, batch: Dict[str, Any], batch_idx: int, dataloader_idx:int=0) -> TensorType["batch", "k"]:
        with torch.no_grad(): 
            z = self(batch["rgb_one"], batch["rgb_two"], batch["g_qpos"]) 
            c = self.C(z) # [batch, k]: prototypes

            return c
        
    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, stage="train")
        
    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:  
        return self._shared_step(batch, stage="val")
        
    def _shared_step(self, batch: Dict[str, Any], stage: str) -> torch.Tensor: 
        losses = []
        
        batch_size, chunk = batch["rgb_one"].shape[:2]
        n = batch_size * chunk
        
        anc = {k: v for k, v in batch.items() if k in self.modalities}
        pos = {k.removesuffix("_pos"): v for k, v in batch.items() if k.endswith("_pos") and k.removesuffix("_pos") in self.modalities}
        
        z_anc = self(**anc) # [n, d_model]: anchor
        z_pos = self(**pos) # [n, d_model]: positive

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
            # Perform sinkhorn on all targets; take only the current ones to perform any downstream tasks
            q = {k: self.sinkhorn(v)[:n] for k, v in c.items()} # [n || n + capacity, k] (each): targets
            
        p = {k: F.softmax(v[:n] / self.prototype_kwargs.tau, dim=-1) for k, v in c.items()} # [n, k]: predictions 
        log_p = {k: F.log_softmax(v[:n] / self.prototype_kwargs.tau, dim=-1) for k, v in c.items()} # [n, k]: predictions 
              
        if self.logger is not None and isinstance(self.logger, pl.loggers.WandbLogger): 
                with torch.no_grad(): 
                    _n, k = q["one"].shape
                    
                    protoype_assignments = q["one"].argmax(dim=-1) # [n]
                    prototype_counts, prototype_bin_edges = np.histogram(protoype_assignments.cpu().numpy(), bins=np.arange(k + 1))
                    prototype_fraction = prototype_counts / _n # [k]

                    prediction_assignments = p["one"].argmax(dim=-1) # [n]
                    prediction_counts, prediction_bin_edges = np.histogram(prediction_assignments.cpu().numpy(), bins=np.arange(k + 1))
                    prediction_fraction = prediction_counts / _n # [k]
                
                    top_prototype_fraction = prototype_fraction.max()
                    top_prediction_fraction = prediction_fraction.max() 
                    
                    prototype_mean = q["one"].float().mean(dim=0) # [k]
                    prototype_entropy = -(prototype_mean*(prototype_mean+1e-8).log()).sum() # []
                    prototype_perplexity_fraction = prototype_entropy.exp()/k # []
                    
                    prediction_mean = p["one"].float().mean(dim=0) # [k]
                    prediction_entropy = -(prediction_mean*(prediction_mean+1e-8).log()).sum() # []
                    prediction_perplexity_fraction = prediction_entropy.exp()/k # []
                    
                    if stage == "train":
                        self.prototype_usage_ema.mul_(self.ema_alpha).add_((1-self.ema_alpha) * torch.tensor(prototype_fraction, device=self.device)) # [k]   
                    # Sum of prototypes whose estimated moving average is smaller than a thenth of the uniform case             
                    num_dead_prototypes = (self.prototype_usage_ema < 0.1/k).float().sum() # []
                    
                    sim = z_anc @ z_anc.T # [n, n]
                    off_diag_similarity = (sim.sum()-sim.diagonal().sum())/(_n*(_n-1))
                    z_std = z_anc_full.std(dim=0).mean()
                                    
                    per_prediction_sample_entropy = (-(p["one"]*log_p["one"])).sum(dim=-1).mean() / math.log(k) # normalized per sample entropy                      
                            
                    self.log_dict({
                        f"{stage}/top_prototype_fraction": top_prototype_fraction, 
                        f"{stage}/top_prediction_fraction": top_prediction_fraction, 
                        f"{stage}/prototype_perplexity_fraction": prototype_perplexity_fraction, 
                        f"{stage}/prediction_perplexity_fraction": prediction_perplexity_fraction, 
                        f"{stage}/num_dead_prototypes": num_dead_prototypes, 
                        f"{stage}/per_prediction_sample_entropy": per_prediction_sample_entropy, 
                        f"{stage}/off_diag_similarity": off_diag_similarity, 
                        f"{stage}/z_std": z_std,             
                       
                    },
                    logger=True, 
                    prog_bar=False,
                    on_step=(stage=="train"),
                    on_epoch=True,
                    sync_dist=True
                    )
                    
                    if stage=="train" and self.global_step % self._log_every_n_steps==0:
                        prototype_histogram = wandb.Histogram(np_histogram=(prototype_counts, prototype_bin_edges))
                        prediction_histogram = wandb.Histogram(np_histogram=(prediction_counts, prediction_bin_edges))

                        self.logger.experiment.log({
                            "train/prototype_histogram": prototype_histogram,
                            "train/prediction_histogram": prediction_histogram, 
                            "trainer/global_step": self.global_step
                        })

        if self.time_contrastive: 
            t_one = z_anc_full[:n].view(batch_size, chunk, -1) # [batch, chunk, k]
            tcn_loss = self.timeContrastiveLoss(t_one) # []
            # losses.extend([tcn_loss])
            
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
            swav_losses[target] = torch.stack([(-(q[prediction] * log_p[target])).sum(dim=-1).mean() for prediction in predictions]).mean()

        losses.extend(swav_losses.values())
    
        if stage == "val":             
            if sum(x.shape[0] for x in self._c_val) < self.num_plot: 
                self._c_val.append(c_one[:n].detach().cpu()) # [n, k]
                self._target_val.append(q["one"][:n].detach().cpu()) # [n, k]
                self._idxs_val.append(batch["idxs"][:, :, 0].reshape(n).cpu()) # [n]: first index of sequence window
                self._task_val.append(batch["task"].reshape(n).cpu()) # [n]
                self._robot_val.append(batch["robot"].reshape(n).cpu()) # [n]: consistent robot over batch 

        loss = torch.stack(losses).mean()
            
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
        rgb_one: Optional[TensorType["batch", "chunk/num_steps", "window", "channels", "height", "width"]]=None, 
        rgb_two: Optional[TensorType["batch", "chunk/num_steps", "window", "channels", "height", "width"]]=None, 
        idxs: Optional[TensorType["batch", "chunk/num_steps", "window"]]=None
        ) -> TensorType["batch*chunk", "d_model"]:
        embs = []

        if "rgb_one" in self.modalities: 
            emb_one = self.lNVisionOne(self.visionEncoder(rgb_one))
            embs.append(emb_one) # [batch*chunk, window, 384] 
        if "rgb_two" in self.modalities: 
            emb_two = self.lNVisionTwo(self.visionEncoder(rgb_two)) # [batch*chunk, (window+1), 384]    
            emb_two = emb_two[:, 1:, ...] - emb_two[:, :-1, ...] # [batch*chunk, window, 384]
            embs.append(emb_two)
        if "g_qpos" in self.modalities: 
            pass 
        
        emb = torch.cat(embs, dim=-1)# [batch*chunk, window, d_model*2]
        emb = self.downEmb(emb) # [batch*chunk, window, d_model]
        emb = self.sequential(emb, idxs=idxs) # [batch*chunk, d_model]
        
        z = F.normalize(emb, dim=-1) # [n, d_model]
            
        return z