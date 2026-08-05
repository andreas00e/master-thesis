# Skill Allignment Transformer 

import os
from omegaconf import DictConfig

import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torchtyping import TensorType

from models.utils import PE
# from models.discover.tse import TSE


class SAT(nn.Module): 
    def __init__(
        self, 
        sat_layer_kwargs: DictConfig, 
        sat_kwargs: DictConfig,   
        skill_encoder_weights: os.PathLike, 
        ) -> None:
        super().__init__() 
        
        self.sat_layer_kwargs = sat_layer_kwargs
        self.sat_kwargs = sat_kwargs
        self.skill_encoder_weights = skill_encoder_weights
        
        self.SAT_layer = nn.TransformerEncoderLayer(**self.sat_layer_kwargs)
        self.SAT = nn.TransformerEncoder(self.SAT_layer, **self.sat_kwargs)
        
        self.skill_encoder_weights = torch.load(self.skill_encoder_weights, weights_only=True) 
        
        self.pe = PE(d_model=self.sat_layer_kwargs.d_model).eval() 
        
            
        self.loss = F.mse_loss()
        
    def forward(self, x: TensorType["*"]) -> TensorType["*"]:
        z_hat = self.SkillEncoder(x)
        
        z = self.pe(x)
        
        loss = self.loss(z, z_hat)
        return loss