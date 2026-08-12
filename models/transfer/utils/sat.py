# Skill Allignment Transformer 

import os
from omegaconf import DictConfig

import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torchtyping import TensorType
from torchvision.models import resnet18

from models.utils.utils import PE
# from models.discover.tse import TSE


class SAT(nn.Module): 
    def __init__(
        self, 
        weights_path: os.PathLike, 
        encoder_layer_kwargs: DictConfig, 
        transformer_encoder_kwargs: DictConfig,
        pe_kwargs: DictConfig,   
        tse_weights: os.PathLike=None, 
        ) -> None:
        super().__init__() 
        
        self.weights_path = weights_path
        self.sat_layer_kwargs = encoder_layer_kwargs
        self.sat_kwargs = transformer_encoder_kwargs
        self.pe_kwargs = pe_kwargs
        self.tse_weights = tse_weights
        
        self.obs_encoder = resnet18(pretrained=False)
        self.linear = nn.Linear(1000, 256)
        
        self.encoder_layer = nn.TransformerEncoderLayer(**self.sat_layer_kwargs)
        self.encoder_transformer = nn.TransformerEncoder(self.encoder_layer, **self.sat_kwargs)
        self.pe = PE(**self.pe_kwargs)
        
        # self.tse_weights = torch.load(self.tse_weights, weights_only=True) 
        self.loss = nn.MSELoss()
        
    def forward(
        self, 
        rgb_obs: TensorType["batch", "steps", "channels", "height", "width"]
        ) -> TensorType["batch", "dim"]:
        
        rgb_obs_shape = rgb_obs.shape
        rgb_obs = rgb_obs.view(-1, *rgb_obs_shape[2:]) # [batch*steps, channels, height, width]
        z_hat = self.obs_encoder(rgb_obs) # [batch*steps, d_model]
        z_hat = z_hat.view(*rgb_obs_shape[:2], -1) # [batch, steps, d_model
        z_hat = self.linear(z_hat) # [1000, 256]
        # z_hat = self.pe(z_hat, )
        z_hat = self.encoder_transformer(z_hat) # [batch, steps, d_model]
        z_tilde = torch.rand_like(z_hat)  # [batch, steps, d_model], TODO: REPLACE WITH TSE EMBEDDINGS!
        
        loss = self.loss(z_tilde, z_hat)
        return loss