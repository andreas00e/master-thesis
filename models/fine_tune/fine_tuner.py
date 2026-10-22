import wandb
from omegaconf import DictConfig
from typing import Any, Dict, Optional, Tuple

import torch
from torch.optim import AdamW
import torch.nn.functional as F
from torchtyping import TensorType
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

import lightning.pytorch as pl

from models.utils.vicreg import VICReg
from models.utils.aux_models import Expander
from models.utils.cnns import Resnet18HalfWidth, plot_feature_maps
        

class FineTunerVisual(pl.LightningModule): 
    def __init__(
        self, 
        viewpoint: str, 
        optimizer_kwargs: DictConfig, 
        scheduler_kwargs: DictConfig, 
        vision_encoder_kwargs: DictConfig, 
        expander_kwargs: DictConfig, 
        vic_reg_kwargs: DictConfig,
        ) -> None: 
        
        super().__init__()
        self.save_hyperparameters() 
        
        self.viewpoint = viewpoint                
        self.optimizer_kwargs = optimizer_kwargs
        self.scheduler_kwargs = scheduler_kwargs
        self.vision_encoder_kwargs = vision_encoder_kwargs 
        self.expander_kwargs = expander_kwargs
        self.vic_reg_kwargs = vic_reg_kwargs
        
        self.visionEncoder = Resnet18HalfWidth(**self.vision_encoder_kwargs)  
        self.visionExpander = Expander(**self.expander_kwargs)
        
        self.vicReg = VICReg(logger=None, **self.vic_reg_kwargs)
                
    def setup(self, stage: Optional[str]=None) -> None:
        if self.logger is not None: 
            self.vicReg.logger = self.logger 

    def _configure_scheduler(
        self, 
        optimizer: Any,   # torch.optim.Optimizer 
        total_steps: int,
        ) -> Any:         # torch.optim.lr_scheduler.LRScheduler
        
        warmup_percentage = self.scheduler_kwargs.get("warmup_percentage", 0.1)
        warmup_steps = max(int(warmup_percentage * total_steps), 1)
        decay_steps = max(total_steps - warmup_steps, 1)
        
        linear_kwargs = dict(self.scheduler_kwargs.linear)
        cosine_kwargs = dict(self.scheduler_kwargs.cosine_annealing)
        linear_kwargs["total_iters"] = warmup_steps
        cosine_kwargs["T_max"] = decay_steps
            
        scheduler_one = LinearLR(optimizer, **linear_kwargs)
        scheduler_two = CosineAnnealingLR(optimizer, **cosine_kwargs)
        scheduler = SequentialLR(optimizer, schedulers=[scheduler_one, scheduler_two],  milestones=[warmup_steps])
        
        return scheduler
                   
    def configure_optimizers(self):
        encoder_parameters = list(filter(lambda p: p.requires_grad, self.visionEncoder.parameters()))
        expander_parameters = list(filter(lambda p: p.requires_grad, self.visionExpander.parameters()))

        param_groups = [
            {
                "params": encoder_parameters, **self.optimizer_kwargs.encoder
                },
            {
                "params": expander_parameters, **self.optimizer_kwargs.expander
                }, 
        ]

        optimizer = AdamW(params=param_groups)
        total_steps = self.trainer.estimated_stepping_batches if self.trainer else 1000

        scheduler = {
            "scheduler": self._configure_scheduler(optimizer, total_steps),
            "interval": "step",
            "frequency": 1,
        }

        return {
            "optimizer": optimizer, 
            "lr_scheduler": scheduler
            }
        
    def on_train_epoch_start(self) -> None: # is only used if consistent_batch==True (so basically never ...)
        if hasattr(self.trainer.train_dataloader, "batch_sampler"):
            if hasattr(self.trainer.train_dataloader.batch_sampler, "set_epoch"):
                self.trainer.train_dataloader.batch_sampler.set_epoch(self.current_epoch)
        
    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, batch_idx, stage="train")
            
    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, batch_idx, stage="val")
    
    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        with torch.no_grad():
            anc_y, pos_y = self(batch[f"rgb_{self.viewpoint}"], batch[f"rgb_{self.viewpoint}_pos"])

            anc_z = self.visionExpander(anc_y)
            pos_z = self.visionExpander(pos_y)

            metrics = {}

            for name, anc, pos in [("encoder", anc_y, pos_y), ("projector", anc_z, pos_z)]:
                anc_norm = F.normalize(anc, dim=-1)
                pos_norm = F.normalize(pos, dim=-1)

                sim = anc_norm @ pos_norm.T
                n = sim.shape[0]
                off_diag = ~torch.eye(n, dtype=torch.bool, device=sim.device)

                sim_same = sim.diagonal().mean()
                sim_diff = sim[off_diag].mean()

                metrics[f"test/{name}/sim_same"] = sim_same
                metrics[f"test/{name}/sim_diff"] = sim_diff
                metrics[f"test/{name}/alignment_margin"] = sim_same - sim_diff

            self.log_dict(
                metrics,
                logger=True,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
        
    def predict_step(self, batch: Dict[str, Any], batch_idx: int) -> Tuple[torch.Tensor, torch.Tensor]: 
        return self((batch["rgb_one"]), (batch["rgb_one_pos"]))
        
    def forward(self, rgb_one: TensorType["*"], rgb_two: TensorType["*"]) -> Tuple[TensorType["n", "d_model"], TensorType["n", "d_model"]]:  
            anc_y_one, anc_y_two, _, anc_y_four, anc_y_out = self.visionEncoder(rgb_one) # [n, d_model]: robot0_eye_in_hand_image 
            pos_y_one, pos_y_two, _, pos_y_four, pos_y_out = self.visionEncoder(rgb_two) # [n, d_model]: agentview_image 
            
            if all([self.training, self.logger is not None, isinstance(self.logger, pl.loggers.WandbLogger), self.global_step % 50 == 0]): 
                anc_y_one_fm = plot_feature_maps(anc_y_one, title="Anchor feature map (layer one)")
                anc_y_two_fm = plot_feature_maps(anc_y_two, title="Anchor feature map (layer two)")
                anc_y_four_fm = plot_feature_maps(anc_y_four, title="Anchor feature map (layer three)")
                pos_y_one_fm = plot_feature_maps(pos_y_one, title="Positive feature map (layer one)")
                pos_y_two_fm = plot_feature_maps(pos_y_two, title="Positive feature map (layer two)")
                pos_y_four_fm = plot_feature_maps(pos_y_four, title="Positive feature map (layer three)")
                
                if isinstance(self.logger, pl.loggers.WandbLogger):
                    self.logger.experiment.log(
                        {   
                            "rgb_one": wandb.Image(rgb_one[0, 0, 0, ...].permute(1, 2, 0).cpu().numpy() * 255.0),
                            "anc_y_one_fm": wandb.Image(anc_y_one_fm), 
                            "anc_y_two_fm": wandb.Image(anc_y_two_fm), 
                            "anc_y_four_fm": wandb.Image(anc_y_four_fm), 
                            "pos_y_three_fm": wandb.Image(pos_y_one_fm), 
                            "pos_y_two_fm": wandb.Image(pos_y_two_fm), 
                            "pos_y_four_fm": wandb.Image(pos_y_four_fm), 
                        }
                    )
                
            return anc_y_out, pos_y_out
        
    def _effective_rank(self, x: torch.Tensor) -> float:
        x_centered = x - x.mean(dim=0)
        s = torch.linalg.svdvals(x_centered.float())
        p = s / (s.sum() + 1e-12)
        entropy = -(p * (p + 1e-12).log()).sum()
        
        return entropy.exp().item()
    
    def _shared_step(self, batch: Dict[str, Any], batch_idx: int, stage: Optional[str]=None) -> torch.Tensor: 
        anc_y, pos_y = self(batch[f"rgb_{self.viewpoint}"], batch[f"rgb_{self.viewpoint}_pos"])
            
        anc_z = self.visionExpander(anc_y) # [n, d_model*x]
        pos_z = self.visionExpander(pos_y) # [n, d_model*x]
        
        if all([stage == "train", self.logger is not None, isinstance(self.logger, pl.loggers.WandbLogger), self.global_step % 50 == 0]): 
            with torch.no_grad(): 
                self.log_dict(
                    {
                        f"{stage}/eff_rank_encoder_anc": self._effective_rank(anc_y),
                        f"{stage}/eff_rank_encoder_pos": self._effective_rank(pos_y),
                        f"{stage}/eff_rank_expander_anc": self._effective_rank(anc_z),
                        f"{stage}/eff_rank_expander_pos": self._effective_rank(pos_z), 
                        f"{stage}/anc_y_std": anc_y.std(dim=0).mean(), 
                        f"{stage}/pos_y_std": pos_y.std(dim=0).mean(), 
                        f"{stage}/anc_z_std": anc_z.std(dim=0).mean(), 
                        f"{stage}/pos_z_std": pos_z.std(dim=0).mean(), 
                        # f"{stage}/anc_z_norm_mean": anc_z.norm(dim=-1).mean(), 
                        # f"{stage}/pos_z_norm_mean": pos_z.norm(dim=-1).mean()
                    },
                    on_step=True, 
                    on_epoch=False,
                    sync_dist=False
                    )
        
        loss, logs_ = self.vicReg(anc_z, pos_z, step=self.global_step if stage == "train" else None)

        self.log_dict(
            {
                f"{stage}/inv_loss": logs_["inv_loss"], 
                f"{stage}/var_loss": logs_["var_loss"],
                f"{stage}/cov_loss": logs_["cov_loss"],
                f"{stage}/tot_loss": loss
            }, 
            logger=True,
            prog_bar=True, 
            on_step=stage=="train", 
            on_epoch=True, 
            sync_dist=True, 
        )

        return loss