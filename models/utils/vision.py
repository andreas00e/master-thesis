from r3m import load_r3m
import loralib as lora 
from pathlib import Path
from omegaconf import  DictConfig
from typing import List, Optional, Union

import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torchtyping import TensorType
import torchvision.models as models

from models.A1_utils.utils import PE


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
        model: str, 
        lora: bool, 
        layer: Union[int, List[int]], 
        r: int, 
        lora_alpha: int, 
        lora_drop_out: float, 
        out_emb_kwargs: DictConfig
        ) -> None:
        super().__init__()
        
        try: 
            self.model = models.get_model(model, weights=None)
        except Exception as e: 
            raise ModuleNotFoundError(f"Model \"{model}\" could not be loaded from torchvision.") from e
        
        try: 
           r3m_model = load_r3m(model)
        except: 
            raise ModuleNotFoundError(f"Model \"r3m_{model}\" could not be loaded from r3m.") from e


        if layer is not None: 
            layer = [layer] if isinstance(layer, int) else list(layer)
            self.layer = [f"layer{i}" for i in layer]
        else: 
            self.layer = []
                
        self.r = r 
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_drop_out
        
        clean_state_dict = {}
        model_state = self.model.state_dict()
        
        for key, value in r3m_model.state_dict().items():
            new_key = key.replace("module.convnav.", "").replace("model.", "")
            if new_key in model_state:
                clean_state_dict[new_key] = value

        self.model.load_state_dict(clean_state_dict, strict=False)
        self.model.fc = nn.Identity()
        
        if lora: 
            self._lora()
            self.out_emb = nn.Linear(**out_emb_kwargs)
        else: 
            self.model.requires_grad_(False)
            self.model.eval()

    def _lora(self) -> None:             
        for block_name, block in self.model.named_children(): 
            if block_name not in self.layer:
                continue 
            
            for _, layer in block.named_children(): 
                for child_name, child in layer.named_children():
                     
                    if isinstance(child, nn.Conv2d): 
                        lora_conv = self._create_lora_conv(child)
                        setattr(layer, child_name, lora_conv)

                    if isinstance(child, nn.Sequential): 
                        for grand_child_name, grand_child in child.named_children(): 
                            if isinstance(grand_child, nn.Conv2d): 
                                lora_conv = self._create_lora_conv(grand_child)
                                setattr(child, grand_child_name, lora_conv)

        lora.mark_only_lora_as_trainable(self.model) 
        
    def _create_lora_conv(self, conv_module: nn.Conv2d) -> lora.ConvLoRA:
        lora_conv = lora.ConvLoRA(
            conv_module=nn.Conv2d, 
            in_channels=conv_module.in_channels, 
            out_channels=conv_module.out_channels, 
            kernel_size=conv_module.kernel_size[0], 
            r=self.r,
            stride=conv_module.stride,
            padding=conv_module.padding, 
            dilation=conv_module.dilation, 
            bias=conv_module.bias is not None, 
            lora_alpha=self.lora_alpha, 
            lora_dropout=self.lora_dropout
        )
        
        with torch.no_grad(): 
            lora_conv.conv.weight.copy_(conv_module.weight)
            if conv_module.bias is not None:  
                lora_conv.conv.bias.copy_(conv_module.bias)
        
        return lora_conv    
        
    def forward(
        self, 
        x: TensorType["batch", "chunk", "window", "channels", "height", "width"]
        ) -> TensorType["batch*chunk*window", "d_model"]:
        
        x_shape = x.shape
        x = x.view(-1, *x.shape[-3:]) # [batch*chunk*window, channels, height, width]
        x = self.model(x) # [batch*chunk*window, feature_dim]
        x = x.view(-1, x.shape[-1]) # [batch*chunk*window, feature_dim]
        x = self.out_emb(x) # [batch*chunk*window, d_model]
        
        return x


class Encoder(nn.Module): 
    def __init__(
        self, 
        encoder_layer_kwargs: DictConfig, 
        transformer_encoder_kwargs: DictConfig, 
        pe_kwargs: DictConfig,
        down_emb_kwargs: Optional[DictConfig]=None, 
        up_emb_kwargs: Optional[DictConfig]=None
        ) -> None:
        
        super().__init__()
            
        encoder_layer = nn.TransformerEncoderLayer(**encoder_layer_kwargs)
        self.encoder_transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer, 
            **transformer_encoder_kwargs
        )
        
        self.is_down_emb = isinstance(down_emb_kwargs, DictConfig)
        self.is_up_emb = isinstance(up_emb_kwargs, DictConfig)
        
        if self.is_down_emb: 
            self.down_emb = nn.Linear(**down_emb_kwargs)
        
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
        
        if self.is_down_emb: 
            x = self.down_emb(x) # [batch*chunk, window, d_model] 
            
        cls = self.cls.expand(x.shape[0], -1, -1) # [batch*chunk, 1, d_model]
        
        x = torch.cat(tensors=(cls, x), dim=1) # [batch*chunk, 1+window, d_model]
        x = self.pe(x, idxs) # [batch*chunk, 1+window, d_model]
        x = self.encoder_transformer(x) # [batch*chunk, 1+window, d_model]
        x = torch.mean(x, dim=1) # [batch*chunk, d_model]
        x = x[:, 0, :] # [batch*chunk, d_model]
        
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
            nn.Linear(in_dim, h1_dim), 
            nn.LayerNorm(h1_dim), 
            nn.GELU(), 
            
            nn.Linear(h1_dim, h2_dim), 
            nn.LayerNorm(h2_dim), 
            nn.GELU(), 
            
            nn.Linear(h2_dim, out_dim)
        ) 
        
    def forward(self, x: TensorType["*"]) -> TensorType["*"]: 
        return self.model(x)          