from omegaconf import DictConfig

import torch
import torch.nn as nn
import torch.nn.functional as F 
from torchtyping import TensorType


class SkillAllignmentTransformer(nn.Module): 
    def __init__(
        self,  
        encoder_layer_kwargs: DictConfig, 
        transformer_encoder_kwargs: DictConfig,
        pe_kwargs: DictConfig,   
        ) -> None:
        super().__init__() 

        self.sat_layer_kwargs = encoder_layer_kwargs
        self.sat_kwargs = transformer_encoder_kwargs
        self.pe_kwargs = pe_kwargs
        self.d_model = encoder_layer_kwargs.d_model
        
        self.x_down = nn.Linear(int(self.d_model*3/2), self.d_model)
                
        self.encoder_layer = nn.TransformerEncoderLayer(**self.sat_layer_kwargs)
        self.encoder_transformer = nn.TransformerEncoder(self.encoder_layer, **self.sat_kwargs)
        
        self.encoder_down = nn.Linear(256, 128)
        
    def forward(
        self, 
        z_tilde: TensorType["batch", "observation_horizon", "d_model"],
        conditions: TensorType["batch", "observation_horizon", "m", "d_model"] 
        ) -> TensorType["batch_size", "dim"]:
        
        conditions = torch.sum(conditions, dim=-2) # [batch, observation_horizon, d_model] 
        z_tilde = z_tilde[:, 1:, :]
        x = torch.cat((z_tilde, conditions), dim=-1) # [batch, observation_horizon, d_model*2] 
        x = self.x_down(x)
        z_hat = self.encoder_transformer(x) # [batch_size, observation_horizon, d_model]
        z_hat = self.encoder_down(z_hat)
        
        loss = F.mse_loss(z_tilde, z_hat)
        return loss