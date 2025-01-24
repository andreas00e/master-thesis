import torch 
import torch.nn as nn 
from torchvision.models import resnet50, ResNet50_Weights 
import torchvision.transforms as T

class ObsEncoderResNet50(nn.Module): 
    def __init__(self, out_dim, *args, **kwargs):
        super().__init__(*args, **kwargs)

        weights = ResNet50_Weights.DEFAULT
        self.layers_pretrained = resnet50(weights=weights)
        self.layers_finetune = nn.Sequential(nn.Linear(1000, out_dim))

    def forward(self, x):
        resize = T.Resize(size=(240, 240))
        x = resize(x)
        return self.layers_pretrained(self.layers_finetune(x))
