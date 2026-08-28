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
        down_emb_kwargs: DictConfig, 
        up_emb_kwargs: DictConfig, 
        pe_kwargs: DictConfig
        ) -> None:
        
        super().__init__()
        
        d_model = encoder_layer_kwargs["d_model"]
        
        self.encoder_layer = nn.TransformerEncoderLayer(**encoder_layer_kwargs)
        self.encoder_transformer = nn.TransformerEncoder(
            encoder_layer=self.encoder_layer, 
            **transformer_encoder_kwargs
        )
        
        self.down_emb = nn.Linear(down_emb_kwargs["in_features"], down_emb_kwargs["out_features"])
        
        self.up_emb = nn.Sequential(
            nn.Linear(up_emb_kwargs["in_features"], up_emb_kwargs["hidden_features"]), 
            nn.ReLU(), 
            nn.Linear(up_emb_kwargs["hidden_features"], up_emb_kwargs["out_features"])
        )
        
        self.pe: nn.Module = PE(**pe_kwargs)
        
        self.cls = nn.Parameter(data=torch.empty(size=(1, 1, d_model), dtype=torch.float32))
        nn.init.xavier_uniform_(self.cls)
    
    def forward(self, x: TensorType["batch*chunk", "window", "d_model"], idxs: TensorType["batch", "chunk", "window"]) -> TensorType["*"]: 
        x = self.down_emb(x) # [batch*chunk, window, d_model] 
        
        cls = self.cls.expand(x.shape[0], -1, -1) # [batch*chunk, 1, d_model]
        x = torch.concat(tensors=(cls, x), dim=1) # [batch*chunk, 1+window, d_model]
        x = self.pe(x, idxs) # [batch*chunk, 1+window, d_model]
        x = self.encoder_transformer(x) # [batch*chunk, 1+window, d_model]
        
        # x = self.up_emb(x[:, 0, :]) # [batch*chunk, 8]
        
        return x 