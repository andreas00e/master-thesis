import os
import math
import wandb
import tempfile
import numpy as np 
import pandas as pd
from functools import partial
from hydra.utils import instantiate
from matplotlib import pyplot as plt 
from typing import Any, Dict, Optional, Union
from omegaconf import DictConfig, OmegaConf

import torch
import torch.nn as nn 
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

import lightning.pytorch as pl 
from torchtyping import TensorType

from models.discover.utils.queue import FIFOQueue
from models.discover.utils.sinkhorn import Sinkhorn
from models.discover.utils.visualize import Visualize
from models.utils.aux_models import TransformerEncoder
from models.utils.cnns import DinoV2
from models.utils.loss import TimeContrastiveLoss, TimeSmoothingLoss
from models.utils.utils import SinusoidalEmbedding


class SkillEncoder(pl.LightningModule): 
    def __init__(
        self, 
        d_model: int, 
        num_plot: int, 
        num_local_chunk: Optional[int], 
        ema_alpha: float, 
        eps: float,
        mag_eps: float,
        mag_momentum_: float, 
        modalities: list[str], 
        queue_start_epochs: int, 
        freeze_c_epochs: int, 
        time_loss: str, 
        temp_time_contrastive: float, 
        visionEncoder_kwargs: DictConfig, 
        sequential_kwargs: DictConfig, 
        prototype_kwargs: DictConfig,  
        optimizer_kwargs: DictConfig, 
        lr_scheduler_kwargs: DictConfig, 
        sinkhorn_kwargs: DictConfig, 
        queue_kwargs: DictConfig, 
        tsne_kwargs: DictConfig, 
        _log_every_n_steps: int=50, 
        ) -> None: 
        
        super().__init__()
        self.save_hyperparameters() 
        
        self.d_model = d_model
        self.num_plot = num_plot
        self.num_local_chunk = num_local_chunk
        self.ema_alpha = ema_alpha
        self.eps = eps
        self.mag_eps = mag_eps
        self.mag_momentum_ = mag_momentum_
        self.modalities = modalities
        self.queue_start_epochs = queue_start_epochs
        self.freeze_c_epochs = freeze_c_epochs
        self.time_loss = time_loss
        self.temp_time_contrastive = temp_time_contrastive
        self.visionEncoder_kwargs = visionEncoder_kwargs
        self.sequential_kwargs = sequential_kwargs
        self.prototype_kwargs = prototype_kwargs
        self.optimizer_kwargs = optimizer_kwargs
        self.lr_scheduler_kwargs = lr_scheduler_kwargs
        self.sinkhorn_kwargs = sinkhorn_kwargs        
        self.queue_kwargs = queue_kwargs
        self.tsne_kwargs = tsne_kwargs
        self._log_every_n_steps = _log_every_n_steps
                
        self.visionEncoder = DinoV2(**self.visionEncoder_kwargs)
        self.visionEncoder.eval() 
        for param in self.visionEncoder.parameters(): 
            param.requires_grad_(False)
                    
        trainable_modules, FiLM_modules = [], []        
        if "rgb_one" in self.modalities: 
            self.visionOneEncoder = nn.Sequential(
                nn.Linear(384, self.d_model),
                nn.ReLU(), 
                nn.Linear(self.d_model, self.d_model), 
                nn.LayerNorm(self.d_model)
            )
            trainable_modules.append(self.visionOneEncoder)
            
            self.FiLMOne = nn.Linear(384, 2 * self.d_model)
            FiLM_modules.append(self.FiLMOne)
            
        if "rgb_two" in self.modalities: 
            self.visionTwoEncoder = nn.Sequential(
                nn.Linear(384, self.d_model),
                nn.ReLU(), 
                nn.Linear(self.d_model, self.d_model), 
                nn.LayerNorm(self.d_model)
            )
            trainable_modules.append(self.visionTwoEncoder)
            
            self.FiLMTwo = nn.Linear(384, 2 * self.d_model)
            FiLM_modules.append(self.FiLMTwo)
            
        if "g_qpos" in self.modalities: 
            self.sinusoidal_embedding = SinusoidalEmbedding(self.d_model)
            self.gripperEncoder = nn.Sequential(
                nn.Linear(self.d_model, self.d_model), 
                nn.ReLU(), 
                nn.Linear(self.d_model, self.d_model), 
                nn.LayerNorm(self.d_model)
            ) 
            self.gripperLN = nn.LayerNorm(self.d_model)
            trainable_modules.append(self.gripperEncoder)
            
            self.FiLMGripper = nn.Linear(384, 2 * self.d_model)
            FiLM_modules.append(self.FiLMGripper)
        
        _features = self.d_model * len([m for m in self.modalities if m != "gripper"])    
        self.downEmb = nn.Sequential(
            nn.Linear(_features, _features), 
            nn.ReLU(),
            nn.Linear(_features, self.d_model), 
            nn.LayerNorm(self.d_model)
            )
        trainable_modules.append(self.downEmb)
        
        self.sequential = TransformerEncoder(**self.sequential_kwargs)
        self.C = nn.Linear(**self.prototype_kwargs.model) # [k, d_model]
        
        if self.time_loss == "contrastive": 
            self.timeLoss = TimeContrastiveLoss(self.temp_time_contrastive)
        elif self.time_loss == "smoothing": 
            self.timeLoss = TimeSmoothingLoss()
        else: 
            self.timeLoss = None
        
        k = self.prototype_kwargs.model.out_features
        self.register_buffer("prototype_usage_ema", torch.full((k, ), 1.0 / k)) # [k]  
        self.register_buffer("mag_mean", torch.zeros(1))
        self.register_buffer("mag_var", torch.ones(1))
        self.register_buffer("mag_init", torch.zeros(1, dtype=torch.bool))
        self.register_buffer("mag_momentum", torch.tensor(self.mag_momentum_))

        self.queue = FIFOQueue(**self.queue_kwargs)
        self.sinkhorn = Sinkhorn(**self.sinkhorn_kwargs)
        self.tsne = Visualize(self.tsne_kwargs)
    
        self._c_val = []
        self._target_val = []
        self._idxs_val = []
        self._task_val = []
        self._robot_val = []
        
        self.init_model_weights(trainable_modules, FiLM_modules)
        
    def init_model_weights(self, trainable_modules, FiLM_modules) -> None:
        for m in trainable_modules:
            m.apply(partial(self._init_weights, mode="trainable"))

        for m in FiLM_modules:
            m.apply(partial(self._init_weights, mode="FiLM"))

        if getattr(self, "C", None) is not None and hasattr(self.C, "weight"):
            nn.init.xavier_uniform_(self.C.weight)
            
    def _init_weights(self, module: nn.Module, mode: str) -> None:
        if isinstance(module, nn.Linear):
            if mode == "trainable":
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif mode == "FiLM":
                nn.init.zeros_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                    with torch.no_grad(): 
                        module.bias[:self.d_model].fill_(1.0)

        elif isinstance(module, nn.LayerNorm):
            if module.weight is not None:
                nn.init.ones_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)   
                
    def configure_optimizers(self) -> Dict[str, Any]:
        trainable_parameters = filter(lambda p: p.requires_grad, self.parameters())
        optimizer = instantiate(self.optimizer_kwargs, params=trainable_parameters)

        linear_kwargs = OmegaConf.to_container(self.lr_scheduler_kwargs.linear, resolve=True)
        cosine_kwargs = OmegaConf.to_container(self.lr_scheduler_kwargs.cosine_annealing, resolve=True)

        warmup_steps = 100
        if hasattr(self, "trainer") and self.trainer is not None and math.isfinite(self.trainer.estimated_stepping_batches):
            total_steps = int(self.trainer.estimated_stepping_batches)
            warmup_percentage = self.lr_scheduler_kwargs.get("warmup_percentage", 0.1)
            warmup_steps = max(1, int(warmup_percentage * total_steps))
            decay_steps = max(1, total_steps - warmup_steps)

            linear_kwargs["total_iters"] = warmup_steps
            cosine_kwargs["T_max"] = decay_steps

        scheduler_one = LinearLR(optimizer, **linear_kwargs)
        scheduler_two = CosineAnnealingLR(optimizer, **cosine_kwargs)
        scheduler = SequentialLR(optimizer, schedulers=[scheduler_one, scheduler_two], milestones=[warmup_steps])

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }
        
    def train(self, mode: bool = True):
        super().train(mode)
        self.visionEncoder.eval()
        return self

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
            
    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, stage="train")
    
    @torch.no_grad()    
    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:  
        return self._shared_step(batch, stage="val")
    
    @torch.no_grad()
    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> None:  
        step_size = 1
        window_size = 16
        
        raw_samples = {k: v.unsqueeze(1) for k, v in batch.items() if k in self.modalities}
        sample = next(iter(raw_samples.values())) # [batch_size, 1, n_steps, ...]
        padding_mask = torch.isnan(sample).all(dim=tuple(range(3, sample.ndim))).squeeze(1) 
        
        samples = {k: torch.nan_to_num(v, nan=0.0) for k, v in raw_samples.items()}      
        
        batch_size, _, n_steps = sample.shape[:3] # [batch_size, 1, n_steps]
        
        z = []
        for s in range(0, n_steps-1, step_size):
            if s+window_size+1 <= n_steps: 
                xs = {k: v[:, :, s:s+window_size+1, ...] for k, v in samples.items()} # [batch_size, 1, step_size, d_model]
                zs = self(padding_mask=padding_mask[:, s:s+window_size], **xs)["global"] # [batch_size, step_size, d_model]
                z.append(zs) 
              
        z = torch.stack(z, dim=1) # [batch_size, n_steps // step_size, d_model]        
        c = self.C(z)  # [batch_size, n_steps // step_size, k]
        c = c.reshape(batch_size, -1, self.prototype_kwargs.model.out_features) # [batch_size, n_steps // step_size, k]
        _, c_top_idxs = torch.topk(c, k=2, dim=-1) # [batch_size, n_steps // step_size, 2]
        c_top_idxs = c_top_idxs.cpu().numpy()
        
        x = np.arange(c_top_idxs.shape[1]) # [n_steps // step_size]
        trajectory_dict = {
            f"trajectory_{i}_top_{j+1}": c_top_idxs[i, :, j] 
            for j in range(1) 
            for i in range(batch_size)
            }
        trajectory_df = pd.DataFrame(trajectory_dict, index=x)
        
        trajectory_df.plot(figsize=(6, 8), grid=True)
        
        plt.xlabel("x")
        plt.ylabel("y")
        plt.show() 
        
    @torch.no_grad()
    def predict_step(self, batch: Dict[str, Any], batch_idx: int, dataloader_idx:int=0) -> TensorType["batch", "k"]:
        with torch.no_grad(): 
            z = self(batch["rgb_one"], batch["rgb_two"], batch["g_qpos"]) 
            c = self.C(z["global"]) # [batch, k]: prototypes

            return c
        
    def _proto_bar(self, vec: torch.Tensor, label: str, value: str, title: str):
        data = [[str(i), float(v)] for i, v in enumerate(vec.detach().float().cpu().tolist())]
        table = wandb.Table(data=data, columns=[label, value])
        return wandb.plot.bar(table, label, value, title=title)
                                
    def _shared_step(self, batch: Dict[str, Any], stage: str) -> torch.Tensor: 
        losses = []
        c = {}
        
        batch_size, chunk = batch[next(iter(batch))].shape[:2] # n=batch_size*chunk
        step = chunk if self.time_loss == "smoothing" else 1
        sinkhorn_step  = batch_size if self.time_loss == "smoothing" else batch_size * chunk
        
        anc = {k: v for k, v in batch.items() if k in self.modalities}
        pos = {k.removesuffix("_pos"): v for k, v in batch.items() if k.endswith("_pos") and k.removesuffix("_pos") in self.modalities}
        
        z_anc = self(**anc) 
        z_pos = self(**pos)
        
        if self.training and "local" in z_anc.keys(): 
            c_anc_local = self.C(z_anc["local"] ) # [n=batch_size*chunk, Optional[num_local_crops], prototypes=k]
            c_pos_local = self.C(z_pos["local"] ) # [n=batch_size*chunk, Optional[num_local_crops], prototypes=k]
            
            c.update({f"anc_local_{i}": c_anc_local[::step, i] for i in range(c_anc_local.shape[1])}) # [n, k] (each)
            c.update({f"pos_local_{i}": c_pos_local[::step, i] for i in range(c_pos_local.shape[1])}) # [n, k] (each)
                        
        c_anc_global = self.C(z_anc["global"]) # [n=batch_size*chunk, k]
        c_pos_global = self.C(z_pos["global"]) # [n=batch_size*chunk, k]

        c.update({"anc_global": c_anc_global[::step], "pos_global": c_pos_global[::step]}) # only use anchor samples for prototypes for temporal smoothing

        with torch.no_grad():
            scores = {"anc_global": c_anc_global[::step].detach(), "pos_global": c_pos_global[::step].detach()} # [n, k] (each)

            if self.current_epoch >= self.queue_start_epochs and self.queue is not None:
                if self.queue.is_full and stage == "train": # only use queue once it is completely filled
                    queue_features = self.queue.get() # get all features
                    scores["anc_global"] = torch.cat([scores["anc_global"], self.C(queue_features[0])], dim=0) # [n+capacity, k]
                    scores["pos_global"] = torch.cat([scores["pos_global"], self.C(queue_features[1])], dim=0) # [n+capacity, k]

                if self.queue is not None and stage == "train":
                    self.queue.enqueue(torch.stack([z_anc["global"][::step].detach(), z_pos["global"][::step].detach()], dim=0))

            # Perform sinkhorn on all targets; take only the current ones to perform any downstream tasks
            q = {k: self.sinkhorn(v)[:sinkhorn_step] for k, v in scores.items()} # [batch_size || batch_size * chunk, k] (each): targets

        p = {k: F.softmax(v / self.prototype_kwargs.tau, dim=-1) for k, v in c.items()} # [n, k]: predictions
        log_p = {k: F.log_softmax(v / self.prototype_kwargs.tau, dim=-1) for k, v in c.items()} # [n, k]: predictions
              
        if self.logger is not None and isinstance(self.logger, pl.loggers.WandbLogger): 
                with torch.no_grad(): 
                    _n, k = q["anc_global"].shape
                    
                    prototype_assignments = q["anc_global"].argmax(dim=-1) # [n]
                    prototype_counts = torch.bincount(prototype_assignments, minlength=k)
                    prototype_fraction = prototype_counts / _n # [k]
                    prototype_fraction_soft = q["anc_global"].sum(dim=0) / q["anc_global"].sum() # [k]

                    prediction_assignments = p["anc_global"].argmax(dim=-1) # [n]
                    prediction_counts = torch.bincount(prediction_assignments, minlength=k)
                    prediction_fraction = prediction_counts / _n # [k]                   
                    prediction_fraction_soft = p["anc_global"].sum(dim=0) / p["anc_global"].sum() # [k]
                
                    top_prototype_fraction = prototype_fraction.max()
                    top_prediction_fraction = prediction_fraction.max() 
                    
                    prototype_mean = q["anc_global"].float().mean(dim=0) # [k]
                    prototype_entropy = -(prototype_mean*(prototype_mean+1e-8).log()).sum() # []
                    prototype_perplexity_fraction = prototype_entropy.exp()/k # []
                    
                    prediction_mean = p["anc_global"].float().mean(dim=0) # [k]
                    prediction_entropy = -(prediction_mean*(prediction_mean+1e-8).log()).sum() # []
                    prediction_perplexity_fraction = prediction_entropy.exp()/k # []
                    
                    if stage == "train":
                        self.prototype_usage_ema.mul_(self.ema_alpha).add_((1-self.ema_alpha) * prototype_fraction.to(self.device))  # [k]   
                    # Sum of prototypes whose estimated moving average is smaller than a thenth of the uniform case             
                    num_dead_prototypes = (self.prototype_usage_ema < 0.1/k).float().sum() # []
                    
                    sim = z_anc["global"][::step] @ z_anc["global"][::step].T # [n, n]
                    off_diag_similarity = (sim.sum() - sim.diagonal().sum()) / (batch_size * (batch_size - 1))
                    z_std = z_anc["global"].std(dim=0).mean()                                    
                    per_prediction_sample_entropy = (-(p["anc_global"] * log_p["anc_global"])).sum(dim=-1).mean() / math.log(k) # normalized per sample entropy                      
                            
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

                if stage == "train" and self.global_step % self._log_every_n_steps == 0:
                    self.logger.experiment.log({
                        "train/prototype_hard_usage": self._proto_bar(prototype_fraction, "prototype_id", "fraction", "Prototype Usage (hard, Sinkhorn targets)"),
                        "train/prototype_soft_usage": self._proto_bar(prototype_fraction_soft, "prototype_id", "fraction", "Prototype Usage (soft, Sinkhorn targets)"),
                        "train/prediction_hard_usage": self._proto_bar(prediction_fraction, "prototype_id", "fraction", "Prototype Usage (hard, predictions)"),
                        "train/prediction_soft_usage": self._proto_bar(prediction_fraction_soft, "prototype_id", "fraction", "Prototype Usage (soft, predictions)"),
                    })
        
        if self.time_loss: 
            t_one = z_anc["global"].reshape(batch_size, chunk, -1) # [batch, chunk, d_model]
            time_loss = self.timeLoss(t_one) # []
            losses.extend([time_loss])
            
            self.log_dict({
                f"{stage}/time_{self.time_loss}_loss": time_loss.detach()
                },
                logger=True, 
                prog_bar=True,
                on_step=(stage == "train"),
                on_epoch=True,
                sync_dist=True
            )

        swav_losses = {}
        for target in q.keys(): # only global "views" are targets (anc_global, pos_global)
            predictions = [k for k in c.keys() if k != target and k.split("_")[0] not in target]
            swav_losses[target] = torch.stack([(-(q[target] * log_p[prediction])).sum(dim=-1).mean() for prediction in predictions]).mean()

        losses.extend(swav_losses.values())
        
        if stage == "val":             
            if sum(x.shape[0] for x in self._c_val) < self.num_plot: 
                self._c_val.append(c["anc_global"].detach().cpu()) # [n, k]
                self._target_val.append(q["anc_global"].detach().cpu()) # [n, k]
                self._idxs_val.append(batch["idxs"][:, ::step, 0].flatten().cpu()) # [n]: first index of sequence window
                self._task_val.append(batch["task"][:, ::step].flatten().cpu()) # [n]
                self._robot_val.append(batch["robot"][:, ::step].flatten().cpu()) # [n]: consistent robot over batch 

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
        rgb_one: Optional[TensorType["batch", "chunk", "window/num_steps", "channels", "height", "width"]]=None, 
        rgb_two: Optional[TensorType["batch", "chunk", "window/num_steps", "channels", "height", "width"]]=None, 
        g_qpos: Optional[TensorType["batch", "chunk", "window/num_steps" "1"]]=None, 
        idxs: Optional[TensorType["batch", "chunk", "window/num_steps"]]=None,
        padding_mask: Optional[TensorType["batch", "chunk", "window/num_steps"]]=None
        ) -> Dict[str, Union[TensorType["batch*chunk", "d_model"], TensorType["batch*chunk", "num_local_chunks", "d_model"]]]:
        embs = []
        z = {}
        
        if "rgb_one" in self.modalities: 
            emb_one = self.visionEncoder(rgb_one) # [batch*chunk, window+1, 384]: DinoV2 embeddings of robot_0_eye_in_hand images
            
            motion_delta_one = emb_one[:, 1:] - emb_one[:, :-1] # [batch*chunk, window, d_model]: temporal differences of robot_0_eye_in_hand embeddings
            gamma_one, beta_one = self.FiLMOne(motion_delta_one).chunk(2, dim=-1) # [batch*chunk, window, d_model] (each)
            
            emb_one = self.visionOneEncoder(emb_one) # [batch*chunk, window+1, d_model]
            emb_one = gamma_one * emb_one[:, :-1] + beta_one # [batch*chunk, window, d_model]                     
            embs.append(emb_one)
            
        if "rgb_two" in self.modalities: 
            emb_two = self.visionEncoder(rgb_two) # [batch*chunk, window+1, 384]: DinoV2 embeddings of agentview images 
            
            motion_delta_two = emb_two[:, 1:] - emb_two[:, :-1] # [batch*chunk, window, d_model]: temporal differences of agentview embeddings
            gamma_two, beta_two = self.FiLMTwo(motion_delta_two).chunk(2, dim=-1) # [batch*chunk, window, d_model] (each)
            
            emb_two = self.visionTwoEncoder(emb_two) # [batch*chunk, window, d_model]
            emb_two = gamma_two * emb_two[:, :-1] + beta_two # [batch*chunk, window, d_model]                     
            embs.append(emb_two) # [batch*chunk, window, d_model]
            
        emb = torch.cat(embs, dim=-1) # [batch*chunk, window, Union[self.d_model, 2 * self.d_model]]
        emb = self.downEmb(emb) # [batch*chunk, window, d_model]
            
        if "g_qpos" in self.modalities: 
            batch_size, chunk = g_qpos.shape[:2]
            emb_g_qpos = self.sinusoidal_embedding(g_qpos.reshape(-1, 1)) # [batch*chunk*window+1, d_model]
            emb_g_qpos = self.gripperEncoder(emb_g_qpos) # [batch*chunk*window+1, d_model]
            emb_g_qpos = emb_g_qpos.reshape(batch_size * chunk, -1, self.d_model)  # [batch*chunk, window+1, d_model]

            motion_delta_gripper = emb_g_qpos[:, 1:] - emb_g_qpos[:, :-1] # [batch*chunk, window+1, d_model]: temporal differences of gripper embeddings
            gamma_two, beta_two = self.FiLMGripper(motion_delta_gripper).chunk(2, dim=-1) # [batch*chunk, window, d_model] (each)
            
            emb = gamma_two * emb + beta_two
        
        emb = self.sequential(emb, idxs=idxs, padding_mask=padding_mask) # [batch*chunk, d_model]
        z_global = F.normalize(emb, dim=-1) # [batch*chunk, d_model]
        z["global"] = z_global
            
        return z