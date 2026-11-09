import os
import math
import wandb
import tempfile
from hydra.utils import instantiate
from typing import Any, Dict, Optional
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
                nn.LayerNorm(self.d_model)
            ) 
            trainable_modules.append(self.gripperEncoder)
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
        x = {k: v.unsqueeze(2) for k, v in batch.items() if k in self.modalities}
        # batch_size, window, chunk = x[self.modalities[0]].shape[:3]
        
        z = self(**x) # [batch_size*n_steps, d_model]
        c = self.C(z) # [batch_size*n_steps, k]
        
        c_top, _ = torch.topk(c, k=2) # [batch_size*n_steps, k]
        
        top_1_c = c_top[:, 0].detach().mean().item()
        top_2_c = c_top[:, 1].detach().mean().item()
            
        self.log("test/top_1_c", top_1_c, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/top_2_c", top_2_c, on_step=False, on_epoch=True, prog_bar=True)
                    
    def _shared_step(self, batch: Dict[str, Any], stage: str) -> torch.Tensor: 
        losses = []
        
        batch_size, chunk = batch["rgb_one"].shape[:2]
        n = batch_size * chunk
        
        anc = {k: v for k, v in batch.items() if k in self.modalities}
        pos = {k.removesuffix("_pos"): v for k, v in batch.items() if k.endswith("_pos") and k.removesuffix("_pos") in self.modalities}
        
        z_anc = self(**anc) # [n, d_model]: anchor
        z_pos = self(**pos) # [n, d_model]: positive

        c_one = self.C(z_anc) # [n, k]
        c_two = self.C(z_pos) # [n, k]
        c = {"one": c_one, "two": c_two}

        with torch.no_grad():
            scores = {"one": c_one.detach(), "two": c_two.detach()} # [n, k] (each)

            if self.current_epoch >= self.queue_start_epochs and self.queue is not None:
                if self.queue.is_full and stage == "train": # only use queue once it is completely filled
                    queue_features = self.queue.get() # get all features
                    scores["one"] = torch.cat([scores["one"], self.C(queue_features[0])], dim=0) # [n + capacity, k]
                    scores["two"] = torch.cat([scores["two"], self.C(queue_features[1])], dim=0) # [n + capacity, k]

                if self.queue is not None and stage == "train":
                    self.queue.enqueue(torch.stack([z_anc.detach(), z_pos.detach()], dim=0))

            # Perform sinkhorn on all targets; take only the current ones to perform any downstream tasks
            q = {k: self.sinkhorn(v)[:n] for k, v in scores.items()} # [n, k] (each): targets

        p = {k: F.softmax(v / self.prototype_kwargs.tau, dim=-1) for k, v in c.items()} # [n, k]: predictions
        log_p = {k: F.log_softmax(v / self.prototype_kwargs.tau, dim=-1) for k, v in c.items()} # [n, k]: predictions
              
        if self.logger is not None and isinstance(self.logger, pl.loggers.WandbLogger): 
                with torch.no_grad(): 
                    _n, k = q["one"].shape
                    
                    prototype_assignments = q["one"].argmax(dim=-1) # [n]
                    prototype_counts = torch.bincount(prototype_assignments, minlength=k)
                    prototype_fraction = prototype_counts / _n # [k]

                    prediction_assignments = p["one"].argmax(dim=-1) # [n]
                    prediction_counts = torch.bincount(prediction_assignments, minlength=k)
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
                        self.prototype_usage_ema.mul_(self.ema_alpha).add_((1-self.ema_alpha) * prototype_fraction.to(self.device))  # [k]   
                    # Sum of prototypes whose estimated moving average is smaller than a thenth of the uniform case             
                    num_dead_prototypes = (self.prototype_usage_ema < 0.1/k).float().sum() # []
                    
                    sim = z_anc @ z_anc.T # [n, n]
                    off_diag_similarity = (sim.sum()-sim.diagonal().sum())/(_n*(_n-1))
                    z_std = z_anc.std(dim=0).mean()                                    
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
            t_one = z_anc.view(batch_size, chunk, -1) # [batch, chunk, d_model]
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
                self._c_val.append(c_one.detach().cpu()) # [n, k]
                self._target_val.append(q["one"].detach().cpu()) # [n, k]
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
            emb_one = self.visionOneLN(self.visionOneEncoder(self.visionEncoder(rgb_one)))
            embs.append(emb_one) # [batch*chunk, window, 384] 
        if "rgb_two" in self.modalities: 
            emb_two = self.visionTwoEncoder(self.visionEncoder(rgb_two)) # [batch*chunk, window+1, 384] 
            emb_two_delta = emb_two[:, 1:] - emb_two[:, :-1] # [batch*chunk, window, 384] 
            log_mag = torch.log(emb_two_delta.norm(dim=-1) + self.eps) # [batch*chunk, window]
            
            if self.training:
                with torch.no_grad():
                    batch_mean, batch_var = log_mag.mean(), log_mag.var() # [], []

                    if not self.mag_init.item():
                        self.mag_mean.copy_(batch_mean)
                        self.mag_var.copy_(batch_var)
                        self.mag_init.fill_(True)
                    else:
                        self.mag_mean.mul_(self.mag_momentum).add_((1 - self.mag_momentum) * batch_mean)
                        self.mag_var.mul_(self.mag_momentum).add_((1 - self.mag_momentum) * batch_var)
                        
            log_mag = ((log_mag - self.mag_mean) / (self.mag_var + self.eps).sqrt()).unsqueeze(-1) # [batch*chunk, window, 1]                        
            emb_two = torch.cat([self.visionTwoLN(emb_two_delta), log_mag], dim=-1) # [batch*chunk, window, self.d_model+1]
            
            embs.append(emb_two) 
            
        emb = torch.cat(embs, dim=-1)# [batch*chunk, window, d_model*2+1]
        emb = self.downEmb(emb) # [batch*chunk, window, d_model]
        emb = self.sequential(emb, idxs=idxs) # [batch*chunk, d_model]
        
        z = F.normalize(emb, dim=-1) # [n, d_model]
            
        return z