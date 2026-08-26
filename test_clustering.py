from omegaconf import DictConfig

import torch 
import torch.nn as nn
from torchtyping import TensorType
from torchvision.models import squeezenet1_1

import lightning.pytorch as pl 

from test_transformer import BottlneckFusion
from models.discover.utils import visionEncoder


class P2OT(pl.LightningModule): 
    def __init__(
        self, 
        fusion_kwargs: DictConfig, 
        ):
        super().__init__()
        
        self.visionEncoder = squeezenet1_1()
        self.inputFusion = BottlneckFusion(**fusion_kwargs)
        
    
    def configure_optimizers(self):
        return super().configure_optimizers()
    
    def forward(self, x):
        return None
    
    def training_step(self, batch, batch_ixs):
        rgb_one = batch["rgb_one"] 
        rgb_two = batch["rgb_two"]
        
        rgb_one_emb = self.visionEncoder(rgb_one)
        rgb_two_emb = self.visionEncoder(rgb_two)
        
        input = [rgb_one_emb, rgb_two]
        output = self.inputFusion(input)
        
        
        
        
        

        return None 
        
    def validation_step(self):
        return None
    
    def test_step(self, *args, **kwargs):
        return None 
    
    
    
    