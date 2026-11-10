import os
import math
import wandb
import tempfile
from hydra.utils import instantiate
from typing import Any, Dict, Optional, Union
from omegaconf import DictConfig, OmegaConf

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
        eps: float,
        mag_eps: float,
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
        self.eps = eps
        self.mag_eps = mag_eps
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
                nn.ReLU(), 
                nn.Linear(self.d_model, self.d_model)
            )
            self.visionOneLN = nn.LayerNorm(self.d_model)
            trainable_modules.extend([self.visionOneEncoder, self.visionOneLN])
            
        if "rgb_two" in self.modalities: 
            self.visionTwoEncoder = nn.Linear(384, self.d_model)
            self.visionTwoLN = nn.LayerNorm(self.d_model)
            trainable_modules.extend([self.visionTwoEncoder, self.visionTwoLN])
            
        if "g_qpos" in self.modalities: 
            self.sinusoidal_embedding = SinusoidalEmbedding(self.d_model)
            self.gripperEncoder = nn.Sequential(
                nn.Linear(self.d_model, self.d_model), 
                nn.ReLU(), 
                nn.Linear(self.d_model, self.d_model), 
            ) 
            self.gripperLN = nn.LayerNorm(self.d_model)
            trainable_modules.extend([self.gripperEncoder, self.gripperLN])
        
        _features = self.d_model * self.num_modalities + (1 if "rgb_two" in self.modalities else 0)       
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
        self.register_buffer("mag_mean", torch.zeros(1))
        self.register_buffer("mag_var", torch.ones(1))
        self.register_buffer("mag_init", torch.zeros(1, dtype=torch.bool))
        self.mag_momentum = 0.99

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
    
    @torch.no_grad()
    def predict_step(self, batch: Dict[str, Any], batch_idx: int, dataloader_idx:int=0) -> TensorType["batch", "k"]:
        with torch.no_grad(): 
            z = self(batch["rgb_one"], batch["rgb_two"], batch["g_qpos"]) 
            c = self.C(z) # [batch, k]: prototypes

            return c
        
    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, stage="train")
    
    @torch.no_grad()    
    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:  
        return self._shared_step(batch, stage="val")
    
    @torch.no_grad()
    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> None:     
        x = {k: v.unfold(1, 8, 4).permute(0, 1, 5, 2, 3, 4) for k, v in batch.items() if k in self.modalities}
        z = self(**x) # [batch_size*n_steps, d_model]
        c = self.C(z) # [batch_size*n_steps, k]
        
        c_top, _ = torch.topk(c, k=2) # [batch_size*n_steps, k]
        
        top_1_c = c_top[:, 0].detach().mean().item()
        top_2_c = c_top[:, 1].detach().mean().item()
            
        self.log("test/top_1_c", top_1_c, on_step=True, on_epoch=True, prog_bar=True)
        self.log("test/top_2_c", top_2_c, on_step=True, on_epoch=True, prog_bar=True)
                    
    def _shared_step(self, batch: Dict[str, Any], stage: str) -> torch.Tensor: 
        losses = []
        c = {}
        
        batch_size, chunk = batch["rgb_one"].shape[:2]
        n = batch_size * chunk
        
        anc = {k: v for k, v in batch.items() if k in self.modalities}
        pos = {k.removesuffix("_pos"): v for k, v in batch.items() if k.endswith("_pos") and k.removesuffix("_pos") in self.modalities}
        
        if self.training: 
            z_anc_local, z_anc_global = self(**anc).values() # [[n, d_model], [n, 4, d_model]]: anchor
            z_pos_local, z_pos_global = self(**pos).values() # [[n, d_model], [n, 4, d_model]]: positive
            
            c_anc_local = self.C(z_anc_local) # [n, 4, k]
            c_pos_local = self.C(z_pos_local) # [n, 4, k]
            
            c.update({f"anc_local_{i}": c_anc_local[:, i] for i in range(c_anc_local.shape[1])})
            c.update({f"pos_local_{i}": c_pos_local[:, i] for i in range(c_pos_local.shape[1])})

        else: 
            z_anc_global = self(**anc)["global"]
            z_pos_global = self(**pos)["global"]
                        
        c_anc_global = self.C(z_anc_global) # [n, k]
        c_pos_global = self.C(z_pos_global) # [n, k]

        c.update({"anc_global": c_anc_global, "pos_global": c_pos_global})

        with torch.no_grad():
            scores = {"anc_global": c_anc_global.detach(), "pos_global": c_pos_global.detach()} # [n, k] (each)

            if self.current_epoch >= self.queue_start_epochs and self.queue is not None:
                if self.queue.is_full and stage == "train": # only use queue once it is completely filled
                    queue_features = self.queue.get() # get all features
                    scores["anc_global"] = torch.cat([scores["anc_global"], self.C(queue_features[0])], dim=0) # [n + capacity, k]
                    scores["pos_global"] = torch.cat([scores["pos_global"], self.C(queue_features[1])], dim=0) # [n + capacity, k]

                if self.queue is not None and stage == "train":
                    self.queue.enqueue(torch.stack([z_anc_global.detach(), z_pos_global.detach()], dim=0))

            # Perform sinkhorn on all targets; take only the current ones to perform any downstream tasks
            q = {k: self.sinkhorn(v)[:n] for k, v in scores.items()} # [n, k] (each): targets

        p = {k: F.softmax(v / self.prototype_kwargs.tau, dim=-1) for k, v in c.items()} # [n, k]: predictions
        log_p = {k: F.log_softmax(v / self.prototype_kwargs.tau, dim=-1) for k, v in c.items()} # [n, k]: predictions
              
        if self.logger is not None and isinstance(self.logger, pl.loggers.WandbLogger): 
                with torch.no_grad(): 
                    _n, k = q["anc_global"].shape
                    
                    prototype_assignments = q["anc_global"].argmax(dim=-1) # [n]
                    prototype_counts = torch.bincount(prototype_assignments, minlength=k)
                    prototype_fraction = prototype_counts / _n # [k]

                    prediction_assignments = p["anc_global"].argmax(dim=-1) # [n]
                    prediction_counts = torch.bincount(prediction_assignments, minlength=k)
                    prediction_fraction = prediction_counts / _n # [k]
                
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
                    
                    sim = z_anc_global @ z_anc_global.T # [n, n]
                    off_diag_similarity = (sim.sum()-sim.diagonal().sum())/(_n*(_n-1))
                    z_std = z_anc_global.std(dim=0).mean()                                    
                    per_prediction_sample_entropy = (-(p["anc_global"]*log_p["anc_global"])).sum(dim=-1).mean() / math.log(k) # normalized per sample entropy                      
                            
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
                        proto_data = [[int(a)] for a in prototype_assignments.cpu().numpy()]
                        prototype_values = wandb.Table(data=proto_data, columns=["prototype_id"])
                        prototype_histogram = wandb.plot.histogram(
                            prototype_values, 
                            value="prototype_id", 
                            title="Prototype Assignment Histogram"
                        )
                        
                        pred_data = [[int(a)] for a in prediction_assignments.cpu().numpy()]
                        prediction_values = wandb.Table(data=pred_data, columns=["prediction_id"])
                        prediction_histogram = wandb.plot.histogram(
                            prediction_values, 
                            value="prediction_id", 
                            title="Prediction Assignment Histogram"
                        )

                        self.logger.experiment.log({
                            "train/prototype_histogram": prototype_histogram,
                            "train/prediction_histogram": prediction_histogram,
                        })

        if self.time_contrastive: 
            t_one = z_anc_global.reshape(batch_size, chunk, -1) # [batch, chunk, d_model]
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
        for target in q.keys(): # only global "views" are targets, anc_global
            predictions = [k for k in c.keys() if k != target and k.split("_")[0] not in target] 
            swav_losses[target] = torch.stack([(-(q[target] * log_p[prediction])).sum(dim=-1).mean() for prediction in predictions]).mean()

        losses.extend(swav_losses.values())
    
        if stage == "val":             
            if sum(x.shape[0] for x in self._c_val) < self.num_plot: 
                self._c_val.append(c_anc_global.detach().cpu()) # [n, k]
                self._target_val.append(q["anc_global"].detach().cpu()) # [n, k]
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
        g_qpos: Optional[TensorType["batch", "chunk/num_steps", "1"]]=None, 
        idxs: Optional[TensorType["batch", "chunk/num_steps", "window"]]=None
        ) -> Dict[str, Union[TensorType["batch*chunk", "d_model"], TensorType["batch*chunk", "4", "d_model"]]]:
        embs = []
        z = {}
        
        if "rgb_one" in self.modalities: 
            if self.training and self.logger is not None and isinstance(self.logger, pl.loggers.WandbLogger) and self.global_step % 50 == 0: 
                self.logger.experiment.log({
                    "rgb_one": wandb.Image(rgb_one[0, 0, 0, ...].permute(1, 2, 0).cpu().numpy() * 255.0)
                })
            
            rgb_one = rgb_one[:, :, :-1, ...] # [batch, chunk, window, channel, height, width]: only 'window' samples
            emb_one = self.visionEncoder(rgb_one) # [batch*chunk*window, 384]
            emb_one = self.visionOneEncoder(emb_one) # [batch*chunk*window, d_model]
            emb_one = self.visionOneLN(emb_one) 
            emb_one = emb_one * self.d_model ** 0.5 
            embs.append(emb_one) 
            
        if "rgb_two" in self.modalities: 
            if self.training and self.logger is not None and isinstance(self.logger, pl.loggers.WandbLogger) and self.global_step % 50 == 0: 
                self.logger.experiment.log({
                    "rgb_two": wandb.Image(rgb_two[0, 0, 0, ...].permute(1, 2, 0).cpu().numpy() * 255.0)
                })
            
            emb_two = self.visionEncoder(rgb_two) # [batch*chunk, window+1, 384] 
            emb_two = self.visionTwoEncoder(emb_two) # [batch*chunk, window+1, self.d_model] 
            emb_two_delta = emb_two[:, 1:] - emb_two[:, :-1] # [batch*chunk, window, d_model] 
            log_mag = torch.log(emb_two_delta.norm(dim=-1) + self.eps) # [batch*chunk, window]
        
            if self.training:
                with torch.no_grad():
                    batch_mean, batch_var = log_mag.mean(), log_mag.var() # [], []

                    if not self.mag_init.item():
                        self.mag_mean.copy_(batch_mean) # []
                        self.mag_var.copy_(batch_var) # []
                        self.mag_init.fill_(True)
                    else:
                        self.mag_mean.mul_(self.mag_momentum).add_((1 - self.mag_momentum) * batch_mean) # []
                        self.mag_var.mul_(self.mag_momentum).add_((1 - self.mag_momentum) * batch_var) # []
                        
            log_mag = ((log_mag - self.mag_mean) / (self.mag_var + self.eps).sqrt()).unsqueeze(-1) # [batch*chunk, window, 1]                        
            emb_two = torch.cat([self.visionTwoLN(emb_two_delta) * self.d_model ** 0.5, self.mag_eps*log_mag], dim=-1) # [batch*chunk, window, self.d_model+1]
            embs.append(emb_two) 
            
        if "g_qpos" in self.modalities: 
            batch, chunk, window = g_qpos.shape[:3]
            emb_g_qpos = self.sinusoidal_embedding(g_qpos.reshape(-1, 1)) # [batch*chunk*window, 1]
            emb_g_qpos = self.gripperEncoder(emb_g_qpos) # [batch*chunk*window, d_model]
            emb_g_qpos = self.gripperLN(emb_g_qpos) * self.d_model ** 0.5 # [batch*chunk*window, d_model]
            emb_g_qpos = emb_g_qpos.reshape(batch*chunk, window, -1) # [batch*chunk, window, d_model]
            embs.append(emb_g_qpos) 
            
        emb = torch.cat(embs, dim=-1) # [batch*chunk, window, ...]
        emb = self.downEmb(emb) # [batch*chunk, window, d_model]
        
        if self.training: 
            window = emb.shape[1]
            subwindow_length = int(window // 4)
            max_start = window - subwindow_length

            starts = torch.randint(0, max_start + 1, (4,), device=emb.device) # [4, ]
            offsets = torch.arange(subwindow_length, device=emb.device) # [subwindow_length]
            indices = starts[:, None] + offsets # [4, subwindow_length]
            
            emb_local = emb.clone()
            emb_local = emb_local[:, indices, :].contiguous() # [batch*chunk, 4, subwindow_length, d_model]
            emb_local = emb_local.reshape(-1, subwindow_length, self.d_model) # [batch*chunk*4, subwindow_length, d_model]            
            emb_local = self.sequential(emb_local) # [batch*chunk*4, d_model]
            
            z_local = F.normalize(emb_local, dim=-1) # [batch*chunk*4, d_model]
            z_local = z_local.reshape(-1, 4, self.d_model) # [batch*chunk, 4, d_model]
            z["local"] = z_local
                
        emb = self.sequential(emb, idxs=idxs) # [batch*chunk, d_model]
        z_global = F.normalize(emb, dim=-1) # [batch*chunk, d_model]
        z["global"] = z_global
            
        return z