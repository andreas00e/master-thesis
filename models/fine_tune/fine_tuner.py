from omegaconf import DictConfig
from hydra.utils import instantiate
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn 
import torch.nn.functional as F 
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

import lightning.pytorch as pl

from models.utils.vicreg import VICReg
from models.utils.aux_models import VisionEncoder, Expander
        

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
        
        self.visionEncoder = VisionEncoder(**self.vision_encoder_kwargs)
        self.visionExpander = Expander(**self.expander_kwargs)
        self.vicReg = VICReg(logger=None, **self.vic_reg_kwargs)
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        
        
    def setup(self, stage: Optional[str]=None) -> None:
        if self.logger is not None: 
            self.vicReg.logger = self.logger 

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
        encoder_parameters = list(filter(lambda p: p.requires_grad, self.visionEncoder.parameters()))
        expander_parameters = list(filter(lambda p: p.requires_grad, self.visionExpander.parameters())) 
        self.visionEncoder.model.convnet.fc.weight.requires_grad = True
        self.visionEncoder.model.convnet.fc.bias.requires_grad = True

        encoder_parameters.extend([self.visionEncoder.model.convnet.fc.weight, self.visionEncoder.model.convnet.fc.bias])
         
        param_groups = [
            {"params": encoder_parameters, **self.optimizer_kwargs.encoder},
            {"params": expander_parameters, **self.optimizer_kwargs.expander}, 
        ]

        optimizer = AdamW(params=param_groups)
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
    
    def test_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        rgb_one = batch["rgb_one"]
        rgb_two = batch["rgb_two"]
        
        rgb_one_ours = self.visionEncoder(rgb_one) # [n, d_model] robot0_eye_in_hand_image 
        rgb_two_ours = self.visionEncoder(rgb_two) # [n, d_model] agentview_image 

        rgb_one_r3m = self.visionEncoder.backbone(rgb_one) 
        rgb_two_r3m = self.visionEncoder.backbone(rgb_one)
        
        rgb_one_ours = F.normalize(rgb_one_ours)
        rgb_two_ours = F.normalize(rgb_two_ours)
        
        rgb_one_r3m = F.normalize(rgb_one_r3m)
        rgb_two_r3m = F.normalize(rgb_two_r3m)

        sim_ours = self.cos(rgb_one_ours, rgb_two_ours)
        sim_r3m = self.cos(rgb_one_r3m, rgb_two_r3m)
        sim_diff = sim_ours - sim_r3m
        
        out = {
            "sim_diff": sim_diff.detach().cpu(), 
        }
        
        return out 
    
    def predict_step(self, batch: Any, batch_idx: int) -> Tuple[torch.Tensor, torch.Tensor]: 
        rgb_one_y = self.visionEncoder(batch["rgb_one"]) # [n, d_model] robot0_eye_in_hand_image 
        rgb_two_y = self.visionEncoder(batch["rgb_two"]) # [n, d_model] agentview_image 
        
        return rgb_one_y, rgb_two_y
    
    def forward(self, batch: Any, batch_idx: int, stage: str) -> torch.Tensor:       
        rgb_one_y = self.visionEncoder(batch["rgb_one"]) # [n, d_model] robot0_eye_in_hand_image 
        rgb_two_y = self.visionEncoder(batch["rgb_two"]) # [n, d_model] agentview_image 
        
        rgb_one_z = self.visionExpander(rgb_one_y) # [n, d_model*x]
        rgb_two_z = self.visionExpander(rgb_two_y) # [n, d_model*x]

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