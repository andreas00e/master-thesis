from omegaconf import DictConfig
from hydra.utils import instantiate
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn 
import torch.nn.functional as F 
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

import lightning.pytorch as pl

from models.fine_tune.fine_tuner import FineTunerVisual
        

class FineTunerGripper(pl.LightningModule): 
    def __init__(
        self, 
        vision_encoder_ckpt: str, 
        optimizer_kwargs: DictConfig, 
        scheduler_kwargs: DictConfig, 
        ) -> None: 
        
        super().__init__()
        self.save_hyperparameters() 
                  
        self.vision_encoder_ckpt = vision_encoder_ckpt 
        self.optimizer_kwargs = optimizer_kwargs
        self.scheduler_kwargs = scheduler_kwargs
        
        fineTunerVisual = FineTunerVisual.load_from_checkpoint(self.vision_encoder_ckpt)
        self.visionEncoder = fineTunerVisual.visionEncoder
        self.visionEncoder.eval()
        
        self.gripperEncoder = nn.Sequential(
                nn.Linear(1, self.d_model // 2), 
                nn.ReLU(), 
                nn.Linear(self.d_model // 2, self.d_model)
            )       
           
        self.strict_loading = False 

    def _configure_scheduler(
        self, 
        optimizer: Any, # torch.optim.Optimizer 
        total_steps: int,
        ) -> Any: # torch.optim.lr_scheduler.LRScheduler
        
        warmup_percentage = self.scheduler_kwargs.get("warmup_percentage", 0.1)
        warmup_steps = int(warmup_percentage * total_steps)
        decay_steps = total_steps - warmup_steps
        
        self.scheduler_kwargs.linear.total_iters = warmup_steps
        self.scheduler_kwargs.cosine_annealing.T_max = decay_steps 
        
        scheduler_one = LinearLR(optimizer, **self.scheduler_kwargs.linear)
        scheduler_two = CosineAnnealingLR(optimizer, **self.scheduler_kwargs.cosine_annealing)
        scheduler = SequentialLR(optimizer, schedulers=[scheduler_one, scheduler_two],  milestones=[warmup_steps])
        
        return scheduler
                   
    def configure_optimizers(self):
        parameters_no_grad = [p.requires_grad_(False) for p in self.visionEncoder.parameters()]        
        parameters = [p for p in self.gripperEncoder.parameters() if p.requires_grad == True]
        
        optimizer = AdamW(params=parameters)
        total_steps = self.trainer.estimated_stepping_batches if self.trainer else 1000

        scheduler = {
            "scheduler": self._configure_scheduler(optimizer, total_steps),
            "interval": "step",
            "frequency": 1,
        }

        return {"optimizer": optimizer, "lr_scheduler": scheduler}
    
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None: 
        trainable_param_names = {name for name, param in self.named_parameters() if param.requires_grad}
        filtered_state_dict = {key: val for key, val in checkpoint["state_dict"].items() if key in trainable_param_names}

        checkpoint["state_dict"] = filtered_state_dict
    
    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        self.strict_loading = False
        
    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        return self(batch, batch_idx, stage="train")
    
    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        return self(batch, batch_idx, stage="val")
    
    def forward(self, batch: Any, batch_idx: int, stage: str) -> torch.Tensor:  
        gripper_emb = self.gripperEncoder(batch["g_qpos"].view(-1, 1))
        rgb_emb = self.visionEncoder(batch["rgb_one"]) # [n, d_model] robot0_eye_in_hand_image 
        
        loss = F.mse_loss(gripper_emb, rgb_emb)
        
        self.log_dict(
            {f"{stage}/mse_loss": loss.detach()},
            logger=True,
            prog_bar=True, 
            on_step=stage=="train", 
            on_epoch=True, 
            sync_dist=True,
        )
        
        return loss