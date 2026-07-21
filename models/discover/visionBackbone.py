import os 

import torch 
import torch.nn as nn 
from torchvision.models import resnet18
from torchtyping import TensorType

import loralib as lora 

from r3m import load_r3m


class VisionBackbone(nn.Module): 
    def __init__(
        self,
        weights_path: os.PathLike     
        ) -> None:
        super().__init__()

        self.weights_path = weights_path
        
        if "r3m" in self.weights_path: 
            self.model = load_r3m
        else: 
            self.model = resnet18(weights=torch.load(f=self.weights_path, weights_only=True))
        
        self._lora()
  
    def _lora(self) -> None: 
        for name_block, block in self.model.named_children(): 
            if name_block not in ["layer3", "layer4"]: 
                continue 
            
            for _, layer in block.named_children(): 
                for child_name, child in layer.named_children(): 
                    if isinstance(child, nn.Conv2d): 
                        lora_conv = lora.ConvLoRA(
                            conv_module = nn.Conv2d,
                            in_channels = child.in_channels, 
                            out_channels = child.out_channels, 
                            kernel_size = child.kernel_size[0], 
                            stride = child.stride,
                            padding = child.padding, 
                            bias = child.bias is not None, 
                            r = 8, 
                            lora_alpha = 16, 
                            lora_dropout = 0
                        )
                        lora_conv.conv.weight.data = child.weight.data.clone()
                        if child.bias is not None:  
                            lora_conv.conv.bias.data = child.bias.data.clone()
                        
                        setattr(layer, child_name, lora_conv)

        lora.mark_only_lora_as_trainable(self.model) 
    
    def forward(self, x) -> TensorType["*"]:
        batch, window = x.shape[:2] 
        x = x.reshape(-1, *x.shape[2:]).permute(0, 3, 1, 2) # [batch*window, channels=3, height, width]
        x = self.model(x) # [batch*window, hidden_dim]
        x = x.reshape(batch, window, -1) # [batch, window, hidden_dim]
        return x