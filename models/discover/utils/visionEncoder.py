from omegaconf import DictConfig 

import torch 
import torch.nn as nn 
import torch.nn.functional as F

from torchtyping import TensorType

from models.utils.utils import PE

class EncoderLayer(nn.TransformerEncoderLayer):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward = 1000,
        dropout = 0.1,
        activation = F.relu,
        layer_norm_eps = 1e-5,
        batch_first = True,
        norm_first = False,
        bias = True,
        device = None,
        dtype = None,
        ) -> None:
        
        super().__init__(
            d_model = d_model,
            nhead = nhead,
            dim_feedforward = dim_feedforward,
            dropout = dropout,
            activation = activation,
            layer_norm_eps = layer_norm_eps,
            batch_first = batch_first,
            norm_first = norm_first,
            bias = bias,
            device = device,
            dtype = dtype,
        )
        
    def forward(self, src, src_mask = None, src_key_padding_mask = None, is_causal = False) -> TensorType["*"]:
        return super().forward(src, src_mask, src_key_padding_mask, is_causal)
        
class TransformerEncoder(nn.TransformerEncoder): 
    def __init__(
        self, 
        encoder_layer, 
        num_layers,
        norm, 
        enable_nested_tensor,
        mask_check,
        ) -> None:
        
        super().__init__(
            encoder_layer = encoder_layer,
            num_layers = num_layers,
            norm = norm,
            enable_nested_tensor = enable_nested_tensor,
            mask_check = mask_check
        )
    
    def forward(self, src, mask = None, src_key_padding_mask = None, is_causal = None) -> TensorType["*"]:
        return super().forward(src, mask, src_key_padding_mask, is_causal)

class VisionEncoder(nn.Module): 
    def __init__(
        self, 
        encoder_layer_kwargs: DictConfig, 
        transformer_encoder_kwargs: DictConfig, 
        out_kwargs: DictConfig, 
        pe_kwargs: DictConfig
        ) -> None:
        
        super().__init__()
        
        self.encoder_layer_kwargs = encoder_layer_kwargs 
        self.transformer_encoder_kwargs = transformer_encoder_kwargs 
        self.out_kwargs = out_kwargs
        self.pe_kwargs = pe_kwargs
        
        self.encoder_layer = EncoderLayer(**self.encoder_layer_kwargs)
        self.encoder_transformer = TransformerEncoder(
            encoder_layer=self.encoder_layer, 
            **self.transformer_encoder_kwargs
            )
        
        self.linear_in = nn.Linear(
            in_features = self.encoder_layer_kwargs.d_model, 
            out_features = self.encoder_layer_kwargs.d_model 
            )
        
        self.linear_out = nn.Sequential(
            nn.Linear(in_features=self.encoder_layer_kwargs.d_model, out_features=self.out_kwargs.hidden_features), 
            nn.ReLU(), 
            nn.Linear(in_features=self.out_kwargs.hidden_features, out_features=self.out_kwargs.out_features)
        )
        
        self.pe = PE(**self.pe_kwargs)

        self.cls = nn.Parameter(data=torch.ones(size=(1, 1, self.encoder_layer_kwargs.d_model)))
    
    def forward(self, x: TensorType["*"], idxs: TensorType["*"]) -> TensorType["*"]: 
        x = self.linear_in(x) # [batch*chunk, window, d_model] 
        x = torch.zeros_like(x) # [batch, chunk, window, d_model]
        
        cls = self.cls.repeat(x.shape[0], 1, 1) # [batch*chunk, 1, d_model]
        x = torch.concat(tensors=(cls, x), dim=1) # [batch*chunk, 1+window, d_model]
        x = self.PE(x, idxs)
        x = self.encoder_transformer(x) # [batch*chunk, 1+window, d_model]
        x = self.linear_out(x[:, 0, :]) # [batch*chunk, 8]
        
        return x 