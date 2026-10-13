from itertools import chain
from omegaconf import DictConfig
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.optim import AdamW
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
        return self(batch, batch_idx, stage="train")
            
    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        return self(batch, batch_idx, stage="val")
    
    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> None: 
        with torch.no_grad(): 
            rgb_one_y = self.visionEncoderOne(batch["rgb_one"]) # [n, d_model]: robot0_eye_in_hand_image 
            rgb_two_y = self.visionEncoderTwo(batch["rgb_two"]) # [n, d_model]: agentview_image
            
            rgb_one_y = F.normalize(rgb_one_y, p=2, dim=-1) # [n, d_model]
            rgb_two_y = F.normalize(rgb_two_y, p=2, dim=-1) # [n, d_model]

            sim = rgb_one_y @ rgb_two_y.T # [n, n]
            sim_same = sim.diagonal().mean().cpu() # []
            
            n = sim.shape[0]
            off_diag = sim[~torch.eye(n, dtype=torch.bool, device=sim.device)]
            sim_diff = off_diag.mean()
            
            self.log_dict(
                {
                    "test/sim_same": sim_same, 
                    "test/sim_diff": sim_diff
                    }, 
                logger=True, 
                on_step=False, 
                on_epoch=True, 
                sync_dist=True  
            )
    
    def predict_step(self, batch: Any, batch_idx: int) -> Tuple[torch.Tensor, torch.Tensor]: 
        rgb_one_y = self.visionEncoderOne(batch["rgb_one"]) # [n, d_model]: robot0_eye_in_hand_image 
        rgb_two_y = self.visionEncoderTwo(batch["rgb_two"]) # [n, d_model]: agentview_image 
        
        return rgb_one_y, rgb_two_y
    
    def forward(self, batch: Any, batch_idx: int, stage: str) -> torch.Tensor:  
        rgb_one_y = self.visionEncoderOne(batch["rgb_one_pos"]) # [n, d_model]: robot0_eye_in_hand_image 
        rgb_two_y = self.visionEncoderTwo(batch["rgb_two_pos"]) # [n, d_model]: agentview_image 
        
        if all([stage == "train", self.logger is not None, isinstance(self.logger, pl.loggers.WandbLogger), self.global_step % 50 == 0]): 
            with torch.no_grad(): 
                y_centered_one = rgb_one_y - rgb_one_y.mean(dim=0) # [n, d_model]
                y_centered_two = rgb_two_y - rgb_two_y.mean(dim=0) 

                S_one = torch.linalg.svdvals(y_centered_one.float()) # [n, n]: singular values
                S_two = torch.linalg.svdvals(y_centered_two.float()) 

                p_one = S_one / S_one.sum() # linear normalization 
                p_two = S_two / S_two.sum() 
                
                shannon_entropy_one = -(p_one * (p_one + 1e-12).log()).sum() # Shannon-Entropy of linearly normalized singular values
                shannon_entropy_two = -(p_two * (p_two + 1e-12).log()).sum() 

                eff_rank_one = shannon_entropy_one.exp() # effective rank of a singular value 
                eff_rank_two = shannon_entropy_two.exp() # effective rank of a singular value 

                self.log_dict(
                    {
                        "train/eff_rank_one": eff_rank_one.cpu().item(),
                        "train/eff_rank_two": eff_rank_two.cpu().item(),
                    },
                    on_step=True, 
                    on_epoch=False,
                    sync_dist=False
                    )
            
        rgb_one_z = self.visionExpanderOne(rgb_one_y) # [n, d_model*x]
        rgb_two_z = self.visionExpanderTwo(rgb_two_y) # [n, d_model*x]
        
        loss, logs_ = self.vicReg(rgb_one_z, rgb_two_z)

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