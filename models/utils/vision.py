from r3m import load_r3m
import loralib as lora 
from pathlib import Path
from  peft import LoraConfig, get_peft_model
from omegaconf import  DictConfig
from typing import List, Optional, Union

import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torchtyping import TensorType

from models.utils.utils import PE


class DepthVisionBackBone(nn.Module): 
    def __init__(
        self
        ) -> None:
        super().__init__()

        self.conv_block_one = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1), 
            nn.BatchNorm2d(32), 
            nn.ReLU(), 
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1), 
            nn.BatchNorm2d(32), 
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.conv_block_two = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1), 
            nn.BatchNorm2d(64), 
            nn.ReLU(), 
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1), 
            nn.BatchNorm2d(64), 
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.fc = nn.Sequential(
            nn.Flatten(), 
            nn.Linear(64*21*21, 512),
            nn.ReLU(), 
            nn.Linear(512, 256), 
        )
    
    def forward(
        self,
        x: TensorType["batch", "chunk", "window", "channels", "height", "width"]
        ) -> TensorType["batch*chunk*window", "d_model"]:
        x_shape = x.shape
        x = x.view(-1, *x.shape[-3:]) # [batch*chunk*window, channels, height, width]
        
        x = self.conv_block_one(x)
        x = self.conv_block_two(x)
        x = self.fc(x)
        
        x = x.view(-1, *x_shape[1:3], x.shape[-1]) # [batch*chunk, window, d_model]
        
        return x
    

class VisionEncoder(nn.Module): 
    def __init__(
        self, 
        model_name: str, 
        d_model: int,
        lora_config_kwargs: DictConfig, 
        ) -> None:
        super().__init__()
        
        self.model_name = model_name 
        self.d_model = d_model  
        self.lora_config_kwargs = lora_config_kwargs
    
        r3m_model = load_r3m(self.model_name)
        backbone = r3m_model.module  
        
        target_modules = [
            f"convnet.layer{k}.{l}.conv{m}" 
            for k in range(4, 5)
            for l in range(1, 2) 
            for m in range(1, 4)
        ]     
        lora_config = LoraConfig(target_modules=target_modules, **lora_config_kwargs)
        
        self.model = get_peft_model(backbone, lora_config)
        
        self.out_emb = nn.Linear(2048, self.d_model)
        
    def train(self, mode: bool=True): 
        # keep BatchNorm2d in eval mode 
        super().train(mode)
        
        if mode: 
            for m in self.model.modules(): 
                if isinstance(m, nn.BatchNorm2d): 
                    m.eval()
                    
        return self
        
    def forward(
        self, 
        x: TensorType["batch", "chunk", "window", "channels", "height", "width"]
        ) -> TensorType["batch*chunk*window", "d_model"]:
        
        x = x.view(-1, *x.shape[-3:]) # [batch*chunk*window, channels, height, width]
        x = self.model(x) # [batch*chunk*window, feature_dim]
        x = self.out_emb(x) # [batch*chunk*window, d_model]
        
        return x
    

class Transformer(nn.Module): 
    def __init__(
        self, 
        encoder_layer_kwargs: DictConfig, 
        transformer_encoder_kwargs: DictConfig, 
        pe_kwargs: DictConfig,
        up_emb_kwargs: Optional[DictConfig]=None
        ) -> None:
        
        super().__init__()
            
        encoder_layer = nn.TransformerEncoderLayer(**encoder_layer_kwargs)
        self.encoder_transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer, 
            **transformer_encoder_kwargs
        )
        
        self.is_up_emb = isinstance(up_emb_kwargs, DictConfig)
        
        if self.is_up_emb:  
            self.up_emb = nn.Sequential(
                nn.Linear(up_emb_kwargs["in_features"], up_emb_kwargs["hidden_features"]), 
                nn.ReLU(), 
                nn.Linear(up_emb_kwargs["hidden_features"], up_emb_kwargs["out_features"])
            )
        
        self.pe = PE(**pe_kwargs)
        self.cls = nn.Parameter(data=torch.empty(size=(1, 1, encoder_layer_kwargs.d_model), dtype=torch.float32))
        nn.init.xavier_uniform_(self.cls)
    
    def forward(
        self, 
        x: TensorType["batch*chunk, window", "d_model"], 
        idxs: Optional[TensorType["batch", "chunk", "window"]]=None
        ) -> TensorType["*"]: 
        
            
        cls = self.cls.expand(x.shape[0], -1, -1) # [batch*chunk, 1, d_model]
        
        x = torch.cat(tensors=(cls, x), dim=1) # [batch*chunk, 1+window, d_model]
        x = self.pe(x, idxs) # [batch*chunk, 1+window, d_model]
        x = self.encoder_transformer(x) # [batch*chunk, 1+window, d_model]
        x = torch.mean(x, dim=1) # [batch*chunk, d_model]
        # x = x[:, 0, :] # [batch*chunk, d_model]
        
        if self.is_up_emb: 
            x = self.up_emb(x) # [batch*chunk, d_model]
        
        return x

class Expander(nn.Module): 
    def __init__(
        self, 
        in_dim: int, 
        h1_dim: int, 
        h2_dim: int, 
        out_dim: int
        ) -> None:
        super().__init__() 
                
        self.model = nn.Sequential(
            nn.Linear(in_features=in_dim, out_features=h1_dim), 
            nn.BatchNorm1d(num_features=h1_dim), 
            nn.GELU(), 
            
            nn.Linear(in_features=h1_dim, out_features=h2_dim), 
            nn.BatchNorm1d(num_features=h2_dim), 
            nn.GELU(), 
            
            nn.Linear(in_features=h2_dim, out_features=out_dim)
        ) 
        
    def forward(self, x: TensorType["*"]) -> TensorType["*"]: 
        return self.model(x)          