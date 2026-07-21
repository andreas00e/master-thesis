import torch 
import torch.nn as nn 
import torch.nn.functional as F

from torchtyping import TensorType

class EncoderLayer(nn.TransformerEncoderLayer):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward = 2048,
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
        norm = None, 
        enable_nested_tensor = True,
        mask_check = True
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
        encoder_layer_kwargs, 
        transformer_encoder_kwargs
        ) -> None:
        super().__init__()
        
        print(encoder_layer_kwargs)
        print(transformer_encoder_kwargs)

        self.encoder_layer_kwargs = encoder_layer_kwargs 
        self.transformer_encoder_kwargs = transformer_encoder_kwargs 
        
        self.encoder_layer = EncoderLayer(**self.encoder_layer_kwargs)
        self.encoder_transformer = TransformerEncoder(
            encoder_layer=self.encoder_layer, 
            **self.transformer_encoder_kwargs
            )
        self.linear_layer = nn.Linear(
            in_features = self.encoder_layer_kwargs.d_model, 
            out_features = self.encoder_layer.linear1.out_features
            )
        
        self.cls = nn.Parameter(data=torch.ones(size=(1, 1, self.encoder_layer.linear1.out_features)), dtype=torch.float32)
    
    def forward(self, x): 
        x = self.linear_layer(x) # [batch, window, hidden_dim] -> [batch, window, d_model]
        self.cls = torch.repeat(self.cls, x.shape[0]) 
        x = torch.concat(tensors=(self.cls, x), dim=1) # [batch, window, d_model] -> [batch, 1 + window, d_model]
        x = self.encoder_transformer(x) # [batch, 1 + window, d_model] -> [batch, 1 + window, d_model]
        x = x[:, 0, :] # [batch, 1 + window, d_model] -> [batch, d_model]
        return x 