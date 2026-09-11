# Skill conditioned Action Decoder (SAD)

import os 
from omegaconf import DictConfig

import torch
import torch.nn as nn
from torchvision.models import resnet18 
from torchtyping import TensorType
import lightning.pytorch as pl

from models.discover.tse import TSE
from models.transfer.utils.sat import SAT
from models.transfer.utils.rce import RCE
from models.transfer.utils.dit import DIT
from models.transfer.utils.pooling import CrossAttentionQueryPooling


class SAD(pl.LightningModule): 
    def __init__(
        self, 
        d_model: int, 
        obs_encoder_ckpt: os.PathLike, 
        tse_ckpt: os.PathLike,
        rce_kwargs: DictConfig,
        dit_kwargs: DictConfig, 
        pool_kwargs: DictConfig, 
        sat_kwargs: DictConfig, 
        optimizer_kwargs: DictConfig
        ) -> None:
        super().__init__()
        self.save_hyperparameters()
       
        self.d_model = d_model
        self.obs_encoder_ckpt = obs_encoder_ckpt
        self.tse_ckpt = tse_ckpt 
        self.rce_kwargs = rce_kwargs
        self.dit_kwargs = dit_kwargs
        self.pool_kwargs = pool_kwargs
        self.sat_kwargs = sat_kwargs
        self.optimizer_kwargs = optimizer_kwargs
        
        # Frozen models
        self.obs_encoder = resnet18(pretrained=False)
        ckpt = torch.load(self.obs_encoder_ckpt, map_location=self.device)
        r3m_state_dict = ckpt.get("state_dict", ckpt)
        
        clean_state_dict = {}
        for key, value in r3m_state_dict.items():
            new_key = key.replace("module.convnav.", "").replace("model.", "")
            if new_key in self.obs_encoder.state_dict():
                clean_state_dict[new_key] = value

        self.obs_encoder.load_state_dict(clean_state_dict, strict=False)
        self.obs_encoder.fc = nn.Identity()
        self.obs_encoder = self.obs_encoder.eval()

        self.tse = None
        
        # Trainable models 
        self.obs_down = nn.Linear(
            int(*list(self.obs_encoder.state_dict().values())[-2].shape), 
            self.d_model
            )
        self.rce = RCE(**self.rce_kwargs)
        self.dit = DIT(**self.dit_kwargs)
        self.sat = SAT(**self.sat_kwargs)
        
        self.pool = CrossAttentionQueryPooling(**self.pool_kwargs)
 
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), **self.optimizer_kwargs.optimizer)
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, **self.optimizer_kwargs.lr_scheduler)
        
        return {
            "optimizer": optimizer, 
            "lr_scheduler": {
                "scheduler": scheduler, 
                "interval": "step"
            }
        }   
    
    def forward(self, batch):
        conditions = []
        actions, rgb_obs, joint_dsc, joint_obs = batch.values() 
        bs, n_steps = actions.shape[:2]        
        
        rgb_obs = rgb_obs.view(-1, *rgb_obs.shape[2:]) # [batch*steps, channels=3, height=224, width=224]
        rgb_emb = self.obs_encoder(rgb_obs) # [batch*steps, *]
        rgb_emb = self.obs_down(rgb_emb) # [batch*steps, d_model]
        rgb_emb = rgb_emb.view(bs, n_steps, -1) # [batch, steps, d_model]
        
        rce_emb = self.rce(joint_dsc, joint_obs) # [batch, steps, d_model]
        
        skl_emb = torch.randn_like(rce_emb) # [batch, steps, d_model]: skill token prototypes 
        # loss_sat = self.sat(rgb_obs)
        
        conditions = [rgb_emb, rce_emb, skl_emb] # list of conditions,,0
        conditions = self.pool(conditions) # [batch, n_steps, k, d_model] 
        
        loss_bc = self.dit(actions, conditions)       
         
        loss_sat = 0
        loss = loss_sat + loss_bc

        stage = self.trainer.state.stage
        self.log_dict({
            f"{stage}_loss_sat": loss_sat, 
            f"{stage}_loss_bc": loss_bc, 
            f"{stage}_loss": loss
        })
        
        return loss 
    
    def _shared_step(self, batch: TensorType["batch"]) -> None: 
        loss = self(batch)
    
        return None
    
    def training_step(self, batch, batch_idx):
        return self._shared_step(batch)
    
    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch)
    
    def test_step(self, batch, batch_idx):
        return self._shared_step(batch)