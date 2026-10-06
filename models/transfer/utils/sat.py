import math
import numpy as np 
from omegaconf import DictConfig

import torch
import torch.nn as nn
import torch.nn.functional as F 
from torchtyping import TensorType

from models.utils.utils import PositionalEncoding


class SkillAllignmentTransformer(nn.Module): 
    def __init__(
        self,  
        num_prototypes: int, 
        encoder_layer_kwargs: DictConfig, 
        transformer_encoder_kwargs: DictConfig,
        pe_kwargs: DictConfig,   
        ) -> None:
        super().__init__() 

        self.num_prototypes = num_prototypes
        self.sat_layer_kwargs = encoder_layer_kwargs
        self.sat_kwargs = transformer_encoder_kwargs
        self.pe_kwargs = pe_kwargs
        self.d_model = encoder_layer_kwargs.d_model
                        
        self.encoder_layer = nn.TransformerEncoderLayer(**self.sat_layer_kwargs)
        self.encoder_transformer = nn.TransformerEncoder(self.encoder_layer, **self.sat_kwargs)
        self.encoder_down = nn.Linear(self.d_model, self.num_prototypes)
        self.obs_down = nn.Linear(self.d_model, self.num_prototypes)
        
        self.positional_encoding = PositionalEncoding(d_model=self.num_prototypes)
        
    def forward(
        self, 
        z_tilde: TensorType["batch", "num_steps", "num_prototypes"],
        conditions: TensorType["batch", "num_steps", "num_modalities", "d_model"] 
        ) -> TensorType["batch_size", "dim"]:
        
        batch_size, num_steps = z_tilde.shape[:2]
        t = np.random.randint(0, num_steps+1, size=(batch_size, )) # [batch]

        conditions = torch.sum(conditions[t], dim=-2) # [batch, d_model] 
        conditions = self.obs_down(conditions) # [batch, num_prototypes] 
        
        x = torch.cat((conditions, z_tilde), dim=1) # [batch, 1+num_steps, num_prototypes] 
        x = x * math.sqrt(self.num_prototypes)
        x = self.positional_encoding(x)
        
        z_hat = self.encoder_transformer(x) # [batch_size, num_steps, num_prototypes]
        z_hat = self.encoder_down(z_hat[:, t+1, :]) # [batch_size, num_prototypes]
        
        loss = F.mse_loss(z_tilde[t], z_hat)
        return loss