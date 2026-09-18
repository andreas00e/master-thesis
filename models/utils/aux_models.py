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



class CNN(nn.Module):

    def __init__(self, out_size) -> None:
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.linear = nn.Linear(6400, out_size)

    def forward(self, images):

        return self.linear(self.cnn(images))


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
        model_name: str="resnet18",  
        d_model: int=512, 
        r: int=64,  
        lora_alpha: int=64, 
        lora_dropout: float=0.05, 
        bias: str="lora_only"
        ) -> None:
        super().__init__()
        
        self.model_name = model_name 
        self.d_model = d_model
        self.r = r 
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.bias = bias 
    
        self.backbone = load_r3m(self.model_name)
        backbone = self.backbone.module  
        
        target_modules = [
            f"convnet.layer{k}.{l}.conv{m}" 
            for k in range(3, 5) # Layers 3 and 4
            for l in range(0, 2) # Blocks 0 and 1
            for m in range(1, 3) # Convlutions 1 and 2
        ]
        
        in_features = 2048 if "50" in self.model_name else 512
        backbone.convnet.fc = nn.Linear(in_features, self.d_model)
        nn.init.xavier_uniform_(backbone.convnet.fc.weight)
        if backbone.convnet.fc.bias is not None: 
            nn.init.zeros_(backbone.convnet.fc.bias)
        
        lora_config = LoraConfig(
            target_modules=target_modules, 
            r=self.r, 
            lora_alpha=self.lora_alpha, 
            lora_dropout=self.lora_dropout, 
            bias = self.bias
            )
        
        self.model = get_peft_model(backbone, lora_config)
        
    def train(self, mode: bool=True): 
        super().train(mode)
        
        if mode: 
            for module in self.model.modules(): 
                if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)): 
                    module.eval() 
                    module.requires_grad_(False)
                    
    def forward(
        self, 
        x: TensorType["batch", "chunk", "window", "channels", "height", "width"]
        ) -> TensorType["batch*chunk*window", "d_model"]:
        
        x = x.view(-1, *x.shape[-3:]) # [batch*chunk*window, channels, height, width]
        x = self.model(x) # [batch*chunk*window, feature_dim]
        
        return x
    
class TransformerEncoder(nn.Module): 
    def __init__(
        self, 
        encoder_layer_kwargs: DictConfig, 
        transformer_encoder_kwargs: DictConfig, 
        pe_kwargs: DictConfig,
        ) -> None:
        super().__init__()
        
        self.encoder_layer_kwargs = encoder_layer_kwargs
        self.transformer_encoder_kwargs = transformer_encoder_kwargs
        self.pe_kwargs = pe_kwargs
        
        self.d_model = int(encoder_layer_kwargs.d_model)
            
        encoder_layer = nn.TransformerEncoderLayer(**self.encoder_layer_kwargs)
        self.transformerEncoder = nn.TransformerEncoder(encoder_layer, **self.transformer_encoder_kwargs)
        
        self.head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model * 2, bias=False),
            nn.BatchNorm1d(self.d_model * 2),
            nn.ReLU(inplace=True), 
            nn.Linear(self.d_model * 2, self.d_model)
            )      
        
        self.cls_token = nn.Parameter(data=torch.empty(size=(1, 1, self.d_model), dtype=torch.float32))
        
        self.pe = PE(**self.pe_kwargs)
        
        self.apply(self._init_weights)
        nn.init.normal_(self.cls_token, std=0.02)

    def _init_weights(self, module): 
        if isinstance(module, nn.Linear): 
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None: 
                nn.init.zeros_(module.bias)       
                
        if isinstance(module, nn.BatchNorm1d): 
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
            
    def forward(
        self, 
        x: TensorType["batch*chunk, window", "d_model"], 
        idxs: Optional[TensorType["batch", "chunk", "window"]]=None
        ) -> torch.Tensor: 
        
        cls_token = self.cls_token.expand(x.shape[0], -1, -1) # [batch*chunk, 1, d_model]
        
        x = torch.cat(tensors=(cls_token, x), dim=1) # [batch*chunk, 1+window, d_model]
        x = self.pe(x, idxs) # [batch*chunk, 1+window, d_model]
        x = self.transformerEncoder(x) # [batch*chunk, 1+window, d_model]
        x = x[:, 0, :] # [batch*chunk, d_model]
        x = self.head(x) # [batch*chunk, d_model]
        
        return x

class Expander(nn.Module): 
    def __init__(
        self, 
        in_dim: int, 
        h_dim: int, 
        out_dim: int
        ) -> None:
        super().__init__() 
        
        self.in_dim = int(in_dim)
        self.h_dim = int(h_dim)
        self.out_dim = int(out_dim)
                
        self.model = nn.Sequential(
            nn.Linear(in_features=self.in_dim, out_features=self.h_dim, bias=None), 
            nn.BatchNorm1d(num_features=self.h_dim), 
            nn.LeakyReLU(negative_slope=0.01), 

            nn.Linear(in_features=self.h_dim, out_features=self.out_dim)
        ) 
        
        self.apply(self._init_weights)
        
    def _init_weights(self, module): 
        if isinstance(module, nn.Linear): 
            nn.init.xavier_uniform_(module.weight, gain=nn.init.calculate_gain("leaky_relu", 0.01))
            if module.bias is not None: 
                nn.init.zeros_(module.bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        return self.model(x)          