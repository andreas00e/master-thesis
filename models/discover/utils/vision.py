import os 
from pathlib import Path
from typing import List, Union
from omegaconf import  DictConfig


import torch 
import torch.nn as nn 
from torchtyping import TensorType

import torchvision.models as models
import loralib as lora 

from models.utils.utils import PE


class VisionBackbone(nn.Module): 
    def __init__(
        self, 
        model: str, 
        weights_path: Union[str, os.PathLike], 
        lora: bool, 
        layer: Union[int, List[int]], 
        r: int, 
        lora_alpha: int, 
        lora_drop_out: float
        ) -> None:
        super().__init__()
        
        try: 
            self.model = models.get_model(model, weights=None)
        except Exception as e: 
            raise ModuleNotFoundError(f"Model \"{model}\" could not be loaded from torchvision.") from e
        
        if weights_path is not None: 
            weights_path = Path(weights_path)
            self.weights_path = weights_path / f"r3m_{model}_weights.pth"
        else: 
            raise ValueError("weights_path cannot be None")
        
        try: 
            ckpt = torch.load(self.weights_path, map_location="cpu")
        except Exception as e: 
            raise FileNotFoundError(f"Weights could not be found at {self.weights_path}.") from e
        
        if layer is not None: 
            layer = [layer] if isinstance(layer, int) else list(layer)
            self.layer = [f"layer{i}" for i in layer]
        else: 
            self.layer = []
        
        self.r = r 
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_drop_out
        
        r3m_state_dict = ckpt.get("state_dict", ckpt)
        clean_state_dict = {}
        model_state = self.model.state_dict()
        for key, value in r3m_state_dict.items():
            new_key = key.replace("module.convnav.", "").replace("model.", "")
            if new_key in model_state:
                clean_state_dict[new_key] = value

        self.model.load_state_dict(clean_state_dict, strict=False)
        self.model.fc = nn.Identity()
          
        if lora: 
            self._lora()
        else: 
            self.model.requires_grad_(False)
            self.model.eval()
  
    def _lora(self) -> None: 
        for name_block, block in self.model.named_children(): 
            if name_block not in self.layer:
                continue 
            
            for _, layer in block.named_children(): 
                for child_name, child in layer.named_children(): 
                    if isinstance(child, nn.Conv2d): 
                        lora_conv = lora.ConvLoRA(
                            conv_module=nn.Conv2d, 
                            in_channels=child.in_channels, 
                            out_channels=child.out_channels, 
                            kernel_size=child.kernel_size[0], 
                            r=self.r,
                            stride=child.stride,
                            padding=child.padding, 
                            dilation=child.dilation, 
                            bias=child.bias is not None, 
                            lora_alpha=self.lora_alpha, 
                            lora_dropout=self.lora_dropout
                        )
                        lora_conv.conv.weight.data = child.weight.data.clone()
                        if child.bias is not None:  
                            lora_conv.conv.bias.data = child.bias.data.clone()
                        
                        setattr(layer, child_name, lora_conv)

        lora.mark_only_lora_as_trainable(self.model) 
    
    def forward(self, x: TensorType["*"]) -> TensorType["*"]:
        return self.model(x)  


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
        
        self.down_emb = nn.Linear(**down_emb_kwargs)
        
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
        x = x[:, 0, :] # [batch*chunk, d_model]
        
        return x