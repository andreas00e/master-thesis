import os 
from omegaconf import DictConfig

import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torchtyping import TensorType

from r3m import load_r3m

from models.utils.utils import PositionalEncoding


class SkillAllignmentTransformer(nn.Module): 
    def __init__(
        self, 
        tse: nn.Module, 
        obs_encoder: nn.Module, 
        encoder_layer_kwargs: DictConfig, 
        transformer_encoder_kwargs: DictConfig,
        pe_kwargs: DictConfig,   
        ) -> None:
        super().__init__() 
        
        self.tse = tse
        self.obs_encoder = obs_encoder
        self.sat_layer_kwargs = encoder_layer_kwargs
        self.sat_kwargs = transformer_encoder_kwargs
        self.pe_kwargs = pe_kwargs
        
        self.linear = nn.Linear(1000, 256)
        
        self.encoder_layer = nn.TransformerEncoderLayer(**self.sat_layer_kwargs)
        self.encoder_transformer = nn.TransformerEncoder(self.encoder_layer, **self.sat_kwargs)
        self.pe = PositionalEncoding(**self.pe_kwargs)
        
    def forward(
        self, 
        item,
        ) -> TensorType["batch_size", "dim"]:
        
        print("Currently in the Skill Alignment Transformer")
        a = self.tse.predict_step(item)
        print(a.shape)
        
        # rgb_obs = rgb_obs.view(-1, *rgb_obs_shape[2:]) # [batch_size*steps, channels, height, width]
        # z_hat = self.obs_encoder(rgb_obs) # [batch_size*steps, d_model]
        # z_hat = z_hat.view(*rgb_obs_shape[:2], -1) # [batch_size, steps, d_model
        # z_hat = self.linear(z_hat) # [1000, 256]
        # # z_hat = self.pe(z_hat, )
        # z_hat = self.encoder_transformer(z_hat) # [batch_size, steps, d_model]
        # z_tilde = torch.rand_like(z_hat)  # [batch_size, steps, d_model], TODO: REPLACE WITH TSE EMBEDDINGS!
        
        # loss = F.mse_loss(z_tilde, z_hat)
        return a