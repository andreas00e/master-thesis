
from typing import Dict, List, Union

import torch
import torch.nn as nn
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers.schedulers import DDPMScheduler, DDIMScheduler

import lightning.pytorch as pl

from RoLD.models.common import SinusoidalPosEmb, get_pe, WrappedTransformerEncoder, WrappedTransformerDecoder, ResBottleneck, ImageAdapter

from models.autoencoder.autoencoder import DownsampleCVAE

class policy(pl.LightningModule): 
    def __init__(self, 
        training_kwargs: Dict[str, Union[int, str, List[int], List[str]]], 
        model_kwargs: Dict,
        noise_scheduler_kwargs: Dict, 
        *args, **kwargs) -> None:
        
        super().__init__()
        self.save_hyperparameters()
        
        self.training_kwargs = training_kwargs, 
        self.model_kwargs = model_kwargs
        self.noise_scheduler_kwargs = noise_scheduler_kwargs
    
        
        self.register_buffer(
            "pe", 
            get_pe(
                hidden_size=self.model_kwargs.hidden_size,
                max_len=self.training_kwargs.window*2
            )
        )
        
        
        self.skill_encoder = None
        self.skill_alignment_transformer = None
        
    def configure_optimizers(self): 
        optimizer = torch.optim.Adam(
            params = [p for p in self.parameters() if p.requires_grad], 
            lr = self.training_wargs.lr
        )
        
        scheduler = get_cosine_schedule_with_warmup(
            optimizer = optimizer, 
            num_warmup_steps = self.training_kwargs.num_warmup_steps, 
            num_training_steps = self.training_kwargs.num_training_steps    
        )
        
        return {
            "optimizer": optimizer, 
            "lr_scheduler": {
                "scheduler": scheduler, 
                "interval": "step"
            }
        }
        
    def _loss(self, inputs, targets): 
        loss = nn.MSELoss()
        return loss(inputs, targets)
        
    def _predict_noise(self): 
        pass
    
    def _predict_epsilon(self, noise, timestep): 
        pass
    
    def _shared_step(self, batch): 
        actions, obs_rgb, obs_depth, states_prob, = batch.values()
        skills_tse = self.skill_encoder(obs_rgb, obs_depth) 
        time_index = torch.randint(
            low = 0, 
            high = self.training_kwargs.num_training_steps, 
        )
        skills_sat = self.skill_alignment_transformer(obs_rgb, obs_depth, skills_tse)
        loss_sat = self._loss(inputs=skills_sat, targets=skills_tse)
        
        pass
    
    def forward(self): 
        pass
        
    def train_step(self): 
        pass
    
    def validation_step(self): 
        pass 
    
    def test_step(self):
        pass