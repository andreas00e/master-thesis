from r3m import load_r3m
from typing import Dict
from omegaconf import DictConfig
from hydra.utils import instantiate

import torch
import lightning.pytorch as pl

from models.utils.vision import VisionEncoder, Expander
from models.utils.vicreg import VICReg


class FineTuner(pl.LightningModule): 
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
        self.vicReg = VICReg(**self.vic_reg_kwargs)
                        
    def configure_optimizers(self) -> Dict:
        trainable_parameters = filter(lambda p: p.requires_grad, self.parameters())        
        optimizer = instantiate(self.optimizer_kwargs, params=trainable_parameters)
        
        scheduler = instantiate(self.scheduler_kwargs, optimizer=optimizer)
        scheduler.T_max = self.trainer.estimated_stepping_batches
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler, 
                "interval": "step"
            }
        }
    
    def on_save_checkpoint(self, checkpoint: Dict) -> None: 
        trainable_param_names = {name for name, param in self.named_parameters() if param.requires_grad}
        filtered_state_dict = {key: val for key, val in checkpoint["state_dict"].items() if key in trainable_param_names}
        
        checkpoint["state_dict"] = filtered_state_dict
    
    def on_load_checkpoint(self, checkpoint: Dict) -> None:
        self.strict_loading = False
        
    def training_step(self, batch, batch_idx) -> torch.Tensor:
        return self(batch, batch_idx, stage="train")
    
    def validation_step(self, batch, batch_idx) -> torch.Tensor:
        return self(batch, batch_idx, stage="val")
    
    def test_step(self, batch, batch_idx) -> torch.Tensor:
        return self(batch, batch_idx, stage="test")
    
    def forward(self, batch, batch_idx, stage) -> torch.Tensor: 
        rgb_one_emb = self.visionEncoder(batch["rgb_one"]) # [n, d_model] robot0_eye_in_hand_image 
        rgb_two_emb = self.visionEncoder(batch["rgb_two"]) # [n, d_model] agentview_image 

        loss, logs = self.vicReg(rgb_two_emb, rgb_one_emb) 
        
        is_train = stage == "train"
        self.log_dict(
            {f"{stage}/loss/{k}": v for k, v in logs.items()}, 
            prog_bar=True,
            on_step=is_train, 
            on_epoch=True, 
            sync_dist=True 
        )

        self.log_dict({f"{stage}/loss": loss}, 
            prog_bar=True,
            on_step=is_train, 
            on_epoch=True, 
            sync_dist=True 
            )

        return loss 