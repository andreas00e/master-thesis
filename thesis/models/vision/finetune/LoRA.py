import r3m 

import torch
import torch.nn as nn
from lora_pytorch import LoRA

# Load pre-trained ResNet18 and remove classification head
resnet = resnet18(pretrained=True)
encoder = nn.Sequential(*list(resnet.children())[:-2])  # Remove avgpool and fc

# Wrap with LoRA (only affects Conv layers by default)
encoder = LoRA.from_module(encoder, rank=4)
