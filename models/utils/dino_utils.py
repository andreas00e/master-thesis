from typing import Optional, Tuple, Union
from omegaconf import DictConfig
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchtyping import TensorType


class ContrastiveHeadDINOv2(nn.Module):
    def __init__(
        self,
        in_channels: int=384,
        cnn_dim:int=128,
        hidden_dim: int=256,
        out_dim:int=128,
        dropout: float=0.1,
        ) -> None:
        super().__init__()
                
        self.in_channels = in_channels
        self.cnn_dim = cnn_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.dropout = dropout
        
        self.mid_channels = max(64, self.cnn_dim // 2)

        self.spatial_net = nn.Sequential(
            nn.Conv2d(self.in_channels, self.mid_channels, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=self.mid_channels),
            nn.GELU(),

            nn.Conv2d(self.mid_channels, self.mid_channels, kernel_size=3, padding=1, groups=self.mid_channels, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=self.mid_channels),
            nn.GELU(),

            nn.Conv2d(self.mid_channels, self.cnn_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(num_groups=16, num_channels=self.cnn_dim),
            nn.GELU(),

            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(), 
            nn.LayerNorm(self.cnn_dim)
        )

        self.cls_norm = nn.LayerNorm(self.in_channels)

        fused_dim = self.cnn_dim + self.in_channels
        self.projector = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(fused_dim, self.hidden_dim, bias=False),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.out_dim, bias=False), 
            nn.LayerNorm(self.out_dim)
        )

    def forward(
        self, 
        patch_tokens: TensorType["batch*chunk*window", "in_channels=384", "6", "6"], 
        cls_token: TensorType["batch*chunk*window", "in_channels=384"]
        ) -> TensorType["batch*chunk*window", "out_dim"]:
        
        spatial_feat = self.spatial_net(patch_tokens) # [batch*chunk*window, cnn_dim]   
        cls_feat = self.cls_norm(cls_token) # [batch*chunk*window, in_channels]   

        fused = torch.cat([spatial_feat, cls_feat], dim=1) # [batch*chunk*window, cnn_dim+in_channels] 
        x = self.projector(fused) # [batch*chunk*window, out_dim]                       
            
        return x

class DinoV2(nn.Module): 
    def __init__(
        self, 
        repo_or_dir: str, 
        model_name: str="vits14", 
        ) -> None:
        super().__init__()
        
        self.model_name = model_name
        self.model = torch.hub.load(
            repo_or_dir=repo_or_dir, 
            model=f"dinov2_{self.model_name}",
            source="local"
            )
        
        self.patch_size = getattr(self.model, "patch_size", 14)
                        
    def forward(
        self,
        x: TensorType["batch", "chunk", "window", "channels", "height", "width"], 
        return_features: bool, 
        ) -> Union[Tuple[TensorType["batch*chunk*window", "384", "6", "6"], TensorType["batch*chunk*window", "384"]], TensorType["batch*chunk*window", "384"]]:
        
        batch_size, chunk, window, _, height, width = x.shape
        x = x.reshape(batch_size*chunk*window, *x.shape[3:]) # [batch*chunk*window, channels, height, width]
                    
        if return_features: 
            x = self.model.forward_features(x)
            
            patch_tokens = x["x_norm_patchtokens"] # [batch*chunk*window, num_patches, feature_dim]
            patch_tokens = patch_tokens.transpose(1, 2).reshape(
                -1, 
                patch_tokens.shape[-1],
                height // self.patch_size,
                width // self.patch_size, 
                ) # [batch*chunk*window, 384, 6, 6]
            
            cls_token = x["x_norm_clstoken"] # [batch*chunk*window, 384]
                        
            return patch_tokens, cls_token
        
        else: 
            x = self.model(x) # [batch*chunk*window, 384]
            x = x.reshape(batch_size*chunk, window, -1) # [batch*chunk, window, out_channels=384]
            
            return x 
        
class DualInputIndependentHeadsModel(nn.Module):
    def __init__(
        self, 
        dino_kwargs: DictConfig, 
        head1_kwargs: DictConfig, 
        head2_kwargs: DictConfig
        ) -> None:
        super().__init__()
        
        self.dino_kwargs = dino_kwargs
        self.head1_kwargs = head1_kwargs
        self.head2_kwargs = head2_kwargs
        
        self.backbone = DinoV2(**self.dino_kwargs)
        self.backbone.eval() 
        for p in self.backbone.parameters(): 
            p.requires_grad_(False)
        
        self.head1 = ContrastiveHeadDINOv2(**self.head1_kwargs)
        self.head2 = ContrastiveHeadDINOv2(**self.head2_kwargs)
        
    def train(self, mode: bool=True):
        super().train(mode)
        self.backbone.eval() 
        
        return self

    def _encode(
        self, 
        x: TensorType["batch", "chunk", "window", "channels", "height", "width"]
        ) -> Tuple[int, int, Tuple[TensorType["batch*chunk", "window", "d_model"]]]:
        
        batch_size, chunk, window = x.shape[:3]
        x = self.backbone(x, return_features=True) # [batch*chunk*window, d_model]
    
        return batch_size*chunk, window, x

    def forward_dino(
        self, 
        x: TensorType["batch", "chunk", "window", "channels", "height", "width"]
        ) -> TensorType["batch*chunk", "window", "d_model"]: 
        
        z = self.backbone(x, return_features=False) # [batch*chunk*window, d_model]
            
        return z 

    def forward_head1(
        self, 
        x1: TensorType["batch", "chunk", "window", "channels", "height", "width"]
        ) -> TensorType["batch*chunk", "window", "d_model"]:
        
        n, window, (patch_grid, cls_token) = self._encode(x1)
        z1 = self.head1(patch_grid, cls_token) # [batch*chunk*window, d_model]
        z1 = z1.reshape(n, window, -1)  # [batch*chunk, window, d_model]
        
        return z1

    def forward_head2(
        self, 
        x2: TensorType["batch", "chunk", "window", "channels", "height", "width"]
        ) -> TensorType["batch*chunk", "window", "d_model"]:
        
        n, window, (patch_grid, cls_token) = self._encode(x2)
        z2 = self.head2(patch_grid, cls_token) # [batch*chunk*window, d_model]
        z2 = z2.reshape(n, window, -1)  # [batch*chunk, window, d_model]
        
        return z2

    def forward(
        self, 
        x1: Optional[TensorType["batch", "chunk", "window", "channels", "height", "width"]]=None, 
        x2: Optional[TensorType["batch", "chunk", "window", "channels", "height", "width"]]=None 
        ) -> Union[TensorType["batch*chunk", "window", "out_dim"], Tuple[TensorType["batch*chunk", "window", "out_dim"], ...], None]:
        
        z1 = self.forward_head1(x1) if x1 is not None else None
        z2 = self.forward_head2(x2) if x2 is not None else None
        
        return z1, z2