from itertools import chain
from omegaconf import DictConfig
from typing import Any, Dict, Optional, Tuple

import torch
from torch.optim import AdamW
import torch.nn.functional as F
from torchtyping import TensorType
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

import lightning.pytorch as pl

from models.utils.vicreg import VICReg
from models.utils.aux_models import CNN, Expander
        

class FineTunerVisual(pl.LightningModule): 
    def __init__(
        self, 
        optimizer_kwargs: DictConfig, 
        scheduler_kwargs: DictConfig, 
        vision_encoder_kwargs: DictConfig, 
        expander_kwargs: DictConfig, 
        vic_reg_kwargs: DictConfig,
        ) -> None: 
        
        super().__init__()
        self.save_hyperparameters() 
                        
        self.optimizer_kwargs = optimizer_kwargs
        self.scheduler_kwargs = scheduler_kwargs
        self.vision_encoder_kwargs = vision_encoder_kwargs 
        self.expander_kwargs = expander_kwargs
        self.vic_reg_kwargs = vic_reg_kwargs
        
        self.visionEncoderOne = CNN(**self.vision_encoder_kwargs)  
        self.visionEncoderTwo = CNN(**self.vision_encoder_kwargs)      
        self.visionExpanderOne = Expander(**self.expander_kwargs)
        self.visionExpanderTwo = Expander(**self.expander_kwargs)
        
        self.vicReg = VICReg(logger=None, **self.vic_reg_kwargs)
                
    def setup(self, stage: Optional[str]=None) -> None:
        if self.logger is not None: 
            self.vicReg.logger = self.logger 

    def _configure_scheduler(
        self, 
        optimizer: Any, # torch.optim.Optimizer 
        total_steps: int,
        ) -> Any: # torch.optim.lr_scheduler.LRScheduler
        
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
        encoder_parameters = list(filter(lambda p: p.requires_grad, chain(self.visionEncoderOne.parameters(), self.visionEncoderTwo.parameters())))
        expander_parameters = list(filter(lambda p: p.requires_grad, chain(self.visionExpanderOne.parameters(), self.visionExpanderTwo.parameters()))) 

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
        
    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, batch_idx, stage="train")
            
    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, batch_idx, stage="val")
    
    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        with torch.no_grad():
            rgb_one_y, rgb_two_y = self(batch["rgb_one"], batch["rgb_two"])

            rgb_one_z = self.visionExpanderOne(rgb_one_y)
            rgb_two_z = self.visionExpanderTwo(rgb_two_y)

            metrics = {}

            for name, one, two in [("encoder", rgb_one_y, rgb_two_y), ("projector", rgb_one_z, rgb_two_z)]:
                one_norm = F.normalize(one, dim=-1)
                two_norm = F.normalize(two, dim=-1)

                sim = one_norm @ two_norm.T
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
        return self((batch["rgb_one"]), (batch["rgb_two"]))
        
    def forward(self, rgb_one: TensorType["*"], rgb_two: TensorType["*"]) -> Tuple[TensorType["n", "d_model"], TensorType["n", "d_model"]]:  
        rgb_one_y = self.visionEncoderOne(rgb_one) # [n, d_model]: robot0_eye_in_hand_image 
        rgb_two_y = self.visionEncoder(rgb_two) # [n, d_model]: agentview_image 
        
        return rgb_one_y, rgb_two_y
    
    def _effective_rank(self, x: torch.Tensor) -> float:
        x_centered = x - x.mean(dim=0)
        s = torch.linalg.svdvals(x_centered.float())
        p = s / (s.sum() + 1e-12)
        entropy = -(p * (p + 1e-12).log()).sum()
        
        return entropy.exp().item()
    
    def _shared_step(self, batch: Dict[str, Any], batch_idx: int, stage: Optional[str]=None) -> torch.Tensor: 
        rgb_one_y, rgb_two_y = self(batch["rgb_one"], batch["rgb_one_pos"])
        
        rgb_one_z = self.visionExpanderOne(rgb_one_y) # [n, d_model*x]
        rgb_two_z = self.visionExpanderTwo(rgb_two_y) # [n, d_model*x]
        
        if all([stage == "train", self.logger is not None, isinstance(self.logger, pl.loggers.WandbLogger), self.global_step % 50 == 0]): 
            with torch.no_grad(): 
                self.log_dict(
                    {
                        f"{stage}/eff_rank_encoder_one": self._effective_rank(rgb_one_y),
                        f"{stage}/eff_rank_encoder_two": self._effective_rank(rgb_two_y),
                        f"{stage}/eff_rank_projector_one": self._effective_rank(rgb_one_z),
                        f"{stage}/eff_rank_projector_two": self._effective_rank(rgb_two_z), 
                        f"{stage}/rgb_one_z_std": rgb_one_z.std(dim=0).mean(), 
                        f"{stage}/rgb_two_z_std": rgb_two_z.std(dim=0).mean(), 
                        f"{stage}/rgb_one_z_norm_mean": rgb_one_z.norm(dim=-1).mean(), 
                        f"{stage}/rgb_two_z_norm_mean": rgb_two_z.norm(dim=-1).mean()
                    },
                    on_step=True, 
                    on_epoch=False,
                    sync_dist=False
                    )
        
        loss, logs_ = self.vicReg(rgb_one_z, rgb_two_z, step=self.global_step if stage == "train" else None)

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