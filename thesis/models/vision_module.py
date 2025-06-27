import torch.nn as nn 
from torchvision.models import resnet50, ResNet50_Weights 


class ObsEncoderResNet50(nn.Module): 
    def __init__(self, out_dim, *args, **kwargs):
        super().__init__(*args, **kwargs)

        weights = ResNet50_Weights.DEFAULT
        self.layers_pretrained = resnet50(weights=weights)
        self.layers_finetune = nn.Sequential(nn.Linear(in_features=1000, out_features=out_dim))

    def forward(self, x):
        return self.layers_finetune(self.layers_pretrained((x)))
