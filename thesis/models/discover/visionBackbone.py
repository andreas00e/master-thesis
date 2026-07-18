import os 

import torch 
import torch.nn as nn 
from torchvision.models import resnet18

from r3m import load_r3m


class VisionBackbone(nn.Module): 
    def __init__(
        self,
        weights_path: os.PathLike,      
        *args, **kwargs
        ) -> None:
        super().__init__()

        self.weights_path = weights_path
        
        if "r3m" in self.weights_path: 
            self.model = load_r3m
        else: 
            self.model = resnet18(weights=torch.load(f=self.weights_path, weights_only=True))
  
    def forward(self, x):
        batch, window = x.shape[:2] 
        x = x.reshape(-1, *x.shape[2:]).permute(0, 3, 1, 2) # [batch*window, channels=3, height, width]
        x = self.model(x) # [batch*window, hidden_dim]
        x = x.reshape(batch, window, -1).mean(dim=1) # [batch, hidden_dim]
        return x