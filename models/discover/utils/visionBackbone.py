import os 
import torch 
import torch.nn as nn 
from torchvision.models import resnet18
from torchtyping import TensorType
import loralib as lora 

class VisionBackbone(nn.Module): 
    def __init__(self, weights_path: os.PathLike) -> None:
        super().__init__()
        self.weights_path = weights_path
        self.model = resnet18(pretrained=False)

        ckpt = torch.load(self.weights_path, map_location="cpu")
        r3m_state_dict = ckpt.get("state_dict", ckpt)
        
        clean_state_dict = {}
        for key, value in r3m_state_dict.items():
            new_key = key.replace("module.convnav.", "").replace("model.", "")
            if new_key in self.model.state_dict():
                clean_state_dict[new_key] = value

        self.model.load_state_dict(clean_state_dict, strict=False)
        self.model.fc = nn.Identity()
            
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
    
    def forward(self, x: torch.Tensor) -> TensorType["*"]:
        x = self.model(x)                  
        return x