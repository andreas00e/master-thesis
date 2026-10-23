import io 
import math
import random 
from PIL import Image
from typing import Dict, Optional
from matplotlib import pyplot as plt

import torch 
import torch.nn as nn 
from torchtyping import TensorType 
from torchvision.models.resnet import BasicBlock


@torch.no_grad()
def plot_feature_maps(
    feature_tensor: TensorType["*"],
    num_plots: int = 16, 
    title: str = "Feature Maps"
) -> Image.Image:
    feature_map = feature_tensor[0].detach().cpu().numpy()
    num_channels = min(num_plots, feature_map.shape[0])
    
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    fig.suptitle(title, fontsize=14)
    
    im = None
    for i, ax in enumerate(axes.flatten()):
        if i < num_channels:
            im = ax.imshow(feature_map[i], cmap="viridis")
            ax.axis("off")
            ax.set_title(f"Ch {i+1}", fontsize=8)
        else:
            ax.axis("off")
            
    if im is not None:
        cbar = fig.colorbar(im, ax=axes.ravel(), orientation='vertical', shrink=0.8)
        cbar.set_label("intensity", fontsize=10)
    
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
    
    pil_img = Image.open(buf).copy() # .copy() verhindert, dass das Bild geschlossen wird, wenn das Byte-Buffer-Objekt verfällt
    return pil_img

@torch.no_grad()
def plot_kernels(
    kernel: TensorType["out_channels", "in_channels", "kernel_height", "kernel_width"],
    num_plots: int = 16, 
    title: str = "Kernel Maps"
    ) -> Image.Image:
    
    kernel = kernel[0].detach().cpu().numpy() # [in_channels, kernel_height, kernel_width]
    num_channels = min(num_plots, kernel.shape[0])
    
    kernels = sorted(random.sample(range(kernel.shape[0]), num_channels))
    
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    fig.suptitle(title, fontsize=14)
    
    im = None
    for i, ax in zip(kernels, axes.flatten()): 
        im = ax.imshow(kernel[i], cmap="viridis")
        ax.axis("off")
        ax.set_title(f"Ch {i+1}", fontsize=8)
        
    for ax in axes.flatten()[num_channels:]:
        ax.axis("off")
        
    if im is not None:
        cbar = fig.colorbar(im, ax=axes.ravel(), orientation="horizontal", shrink=0.8)
        cbar.set_label("Kernel weights", fontsize=10) 
    
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
    
    pil_img = Image.open(buf).copy()
    return pil_img
    
class SpatialSoftmax(nn.Module):
    def __init__(
        self, 
        size: int
        ) -> None:
        super().__init__()
        
        self.size = size
        
        pos_y, pos_x = torch.meshgrid(
            torch.linspace(-1.0, 1.0, self.size),
            torch.linspace(-1.0, 1.0, self.size),
            indexing="ij"
        )
        
        self.register_buffer("pos_x", pos_x.reshape(1, 1, self.size, self.size))
        self.register_buffer("pos_y", pos_y.reshape(1, 1, self.size, self.size))
        
        self.softmax2d = nn.Softmax2d()

    def forward(self, x: TensorType["batch", "channel", "height", "width"]) -> TensorType["batch", "channel*2"]:        
        x = self.softmax2d(x)
        
        expected_x = torch.sum(x * self.pos_x, dim=(-2, -1)) # [batch, channel]
        expected_y = torch.sum(x * self.pos_y, dim=(-2, -1)) # [batch, channel]
        
        feature_vector = torch.hstack([expected_x, expected_y]) # [batch, channel*2]
        
        return feature_vector
    
class BuildAResnet18(nn.Module): 
    def __init__(
        self, 
        width: float, 
        out_dim: int, 
        num_layers: int, 
        what_to_plot: Optional[str]=None 
        ) -> None:
        super().__init__()
        
        if not (width in [0.25, 0.5, 0.75, 1.0]): 
            raise ValueError(f"The width fraction has to be in [0.25, 0.5, 0.75, 1.0], got {width}")
        if not (1 <= num_layers <= 4):            
            raise ValueError(f"The number of layers has to be between 1 and 4, got {num_layers}")
        
        self.width = width
        self.feature_dim = out_dim
        self.num_layers = num_layers 
        self.what_to_plot = what_to_plot
        self.inplanes = int(64 * self.width)
        
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=int(64 * self.width), kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(int(64 * self.width))
        self.relu = nn.ReLU(inplace=True)
        
        for i in range(1, self.num_layers+1): # [32, 84, 84], [64, 42, 42], [128, 21, 21], [256, 11, 11]
            stride = 1 if i == 1 else 2
            setattr(self, f"layer{i}", self._make_layer(BasicBlock, int((32 * 2 ** i) * self.width), 2, stride=stride))
            
        size = int(math.ceil(84/2**(self.num_layers-1)))
        self.spatialSoftmax = SpatialSoftmax(size=size)
        
        num_channels = int((32 * 2 ** self.num_layers) * self.width)
        self.fc = nn.Linear(num_channels * 2, self.feature_dim)
        
    def _make_layer(self, block: BasicBlock, planes: int, blocks: int, stride: int = 1) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
        
    def forward(
        self, 
        x: TensorType["batch", "window", "chunk", "channel", "height", "width"]
        ) -> Dict[str, TensorType["*"]]: 
        out = {}
        
        x = x.reshape(-1, *x.shape[-3:])
        x = self.relu(self.bn1(self.conv1(x)))

        for i in range(1, self.num_layers+1): 
            layer = getattr(self, f"layer{i}")
            x = layer(x)
            
            if self.what_to_plot == "kernels": 
                out[f"kernel{i}"] = layer[0].conv2.weight
            elif self.what_to_plot == "feature_maps": 
                out[f"feature_map{i}"] = x
        
        x = self.spatialSoftmax(x)
        out["x"] = self.fc(x)
        
        return out