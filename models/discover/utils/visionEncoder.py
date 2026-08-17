from omegaconf import DictConfig, OmegaConf

import torch 
import torch.nn as nn 
from torchtyping import TensorType


from models.utils.utils import PE


class VisionEncoder(nn.Module): 
    def __init__(
        self, 
        encoder_layer_kwargs: DictConfig, 
        transformer_encoder_kwargs: DictConfig, 
        out_kwargs: DictConfig, 
        pe_kwargs: DictConfig
        ) -> None:
        
        super().__init__()
        
        encoder_layer_dict = OmegaConf.to_container(encoder_layer_kwargs, resolve=True) if isinstance(encoder_layer_kwargs, DictConfig) else encoder_layer_kwargs
        transformer_encoder_dict = OmegaConf.to_container(transformer_encoder_kwargs, resolve=True) if isinstance(transformer_encoder_kwargs, DictConfig) else transformer_encoder_kwargs
        out_dict = OmegaConf.to_container(out_kwargs, resolve=True) if isinstance(out_kwargs, DictConfig) else out_kwargs
        pe_dict = OmegaConf.to_container(pe_kwargs, resolve=True) if isinstance(pe_kwargs, DictConfig) else pe_kwargs

        self.d_model = encoder_layer_dict["d_model"]
        self.out_kwargs = out_dict
        
        self.encoder_layer = nn.TransformerEncoderLayer(**encoder_layer_dict)
        self.encoder_transformer = nn.TransformerEncoder(
            encoder_layer=self.encoder_layer, 
            **transformer_encoder_dict
        )
        
        self.linear_in = nn.Linear(
            in_features=512, 
            out_features=self.d_model
        )
        
        self.linear_out = nn.Sequential(
            nn.Linear(in_features=self.d_model, out_features=self.out_kwargs["hidden_features"]), 
            nn.ReLU(), 
            nn.Linear(in_features=self.out_kwargs["hidden_features"], out_features=self.out_kwargs["out_features"])
        )
        
        self.pe = PE(**pe_dict)
        self.cls = nn.Parameter(data=torch.ones(size=(1, 1, self.d_model)))
    
    def forward(self, x: TensorType["batch*chunk", "window", "*"], idxs: TensorType["batch", "chunk", "window"]) -> TensorType["*"]: 
        x = self.linear_in(x) # [batch*chunk, window, d_model] 
        
        cls = self.cls.expand(x.shape[0], -1, -1) # [batch*chunk, 1, d_model]
        x = torch.concat(tensors=(cls, x), dim=1) # [batch*chunk, 1+window, d_model]
        x = self.pe(x, idxs) # [batch*chunk, 1+window, d_model]
        x = self.encoder_transformer(x) # [batch*chunk, 1+window, d_model]
        x = self.linear_out(x[:, 0, :]) # [batch*chunk, 8]
        
        return x 