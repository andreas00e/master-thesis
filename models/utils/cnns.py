import io 
from PIL import Image
from typing import Tuple, Union
from matplotlib import pyplot as plt

import torch 
import torch.nn as nn 
from torchtyping import TensorType 
from torchvision.models.resnet import resnet18, BasicBlock


@torch.no_grad()
def plot_feature_maps(
    feature_tensor: TensorType["*"],
    num_plots: int = 16, 
    title: str = "Feature Maps"
) -> Image.Image:
    """Plots feature maps from a tensor and returns a PIL Image for logging."""
    feature_map = feature_tensor[0].detach().cpu().numpy()
    num_channels = min(num_plots, feature_map.shape[0])
    
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    fig.suptitle(title, fontsize=14)
    
    for i, ax in enumerate(axes.flatten()):
        if i < num_channels:
            ax.imshow(feature_map[i], cmap="viridis")
            ax.axis("off")
            ax.set_title(f"Ch {i+1}", fontsize=8)
        else:
            ax.axis("off")
            
    plt.tight_layout()
    
    # Save figure to an in-memory buffer and load as a PIL Image
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
    
    pil_img = Image.open(buf)
    return pil_img

class ThreeLayerCNN(nn.Module):
    def __init__(
        self, 
        feature_dim: int=64,
        negative_slope: float=0.02, 
        with_vis: bool=False, 
        ) -> None:
        super().__init__()
        
        self.feature_dim = feature_dim
        self.negative_slope = negative_slope
        self.with_vis = with_vis
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.LeakyReLU(negative_slope=self.negative_slope)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2) # 32x42x42
        
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.LeakyReLU(negative_slope=self.negative_slope)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2) # 64x21x21
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.LeakyReLU(negative_slope=self.negative_slope)
        self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2) # 64x10x10
        
        self.spatialSoftmax = SpatialSoftmax(height=10, width=10)

        self.fc = nn.Sequential(
            nn.Linear(128 * 2, self.feature_dim),
            nn.LeakyReLU(),
            nn.Linear(self.feature_dim, self.feature_dim), 
            nn.LeakyReLU(),
            nn.Linear(self.feature_dim, self.feature_dim)
        )

    def forward(
        self, 
        x: TensorType["batch", "feature_dim"]
        ) -> Union[Tuple[TensorType["batch", "feature_dim"], ...], TensorType["batch", "feature_dim"]]:
        x = x.reshape(-1, *x.shape[-3:])
        
        out1 = self.pool1(self.relu1(self.conv1(x)))
        out2 = self.pool2(self.relu2(self.conv2(out1)))
        out3 = self.pool3(self.relu3(self.conv3(out2)))
        out4 = self.fc(self.spatialSoftmax(out3))
        
        if self.with_vis: 
            return out1, out2, out3, out4
        else: 
            return out4
        
class Resnet18HalfWidth(nn.Module): 
    def __init__(self, feature_dim: int, with_vis: bool) -> None:
        super().__init__()
        
        self.feature_dim = feature_dim
        self.with_vis = with_vis
        self.inplanes = 32
        
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
                
        self.layer1 = self._make_layer(BasicBlock, 32, 2)            # [batch, 32, 84, 84]
        self.layer2 = self._make_layer(BasicBlock, 64, 2, stride=2)  # [batch, 64, 42, 42]
        self.layer3 = self._make_layer(BasicBlock, 128, 2, stride=2) # [batch, 128, 21, 21]
        self.layer4 = self._make_layer(BasicBlock, 256, 2, stride=2) # [batch, 256, 11, 11]
        
        self.spatialSoftmax = SpatialSoftmax(height=11, width=11)
        self.fc = nn.Linear(256 * 2, self.feature_dim)
        
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
        x: TensorType["batch", "window", "chunk", "height", "width", "channel"]
        ) -> TensorType["batch*window*chunk", "feature_dim"]: 
        x = x.reshape(-1, *x.shape[-3:])
        x = self.relu(self.bn1(self.conv1(x)))
        
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        
        out = self.spatialSoftmax(out4)
        out = self.fc(out)
        
        if self.with_vis: 
            return out1, out2, out3, out4, out 
        else: 
            return out 
        
class Resnet18(nn.Module): 
    def __init__(self, feature_dim: int, with_vis: bool) -> None:
        super().__init__()
        
        self.feature_dim = feature_dim
        self.with_vis = with_vis

        self.model = resnet18() 
        self.model.conv1 = nn.Conv2d(
            in_channels=3, 
            out_channels=64, 
            kernel_size=3, 
            stride=1,
            padding=1,
            bias=False
        )
        self.model.fc = nn.Linear(1024, self.feature_dim)
        
        self.spatialSoftmax = SpatialSoftmax(height=11, width=11)
                    
    def forward(
        self, 
        x: TensorType["batch", "channel", "height", "width"]
        ) -> Union[Tuple[TensorType["batch", "feature_dim"], ...], TensorType["batch", "feature_dim"]]:
        x = x.reshape(-1, *x.shape[-3:])
                
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        
        out1 = self.model.layer1(x)
        out2 = self.model.layer2(out1)
        out3 = self.model.layer3(out2)
        out4 = self.model.layer4(out3)
        
        out = self.spatialSoftmax(out4)
        
        out = self.model.fc(out)
        
        if self.with_vis: 
            return out1, out2, out3, out4, out
        else: 
            return out
        
        
class SpatialSoftmax(nn.Module):
    def __init__(
        self, 
        height: int,
        width: int, 
        ) -> None:
        super().__init__()
        
        self.height = height
        self.width = width
        
        pos_y, pos_x = torch.meshgrid(
            torch.linspace(-1.0, 1.0, self.height),
            torch.linspace(-1.0, 1.0, self.width),
            indexing="ij"
        )
        
        self.register_buffer("pos_x", pos_x.reshape(1, 1, self.height, self.width))
        self.register_buffer("pos_y", pos_y.reshape(1, 1, self.height, self.width))
        
        self.softmax2d = nn.Softmax2d()

    def forward(self, x: TensorType["batch", "channel", "height", "width"]) -> TensorType["batch", "channel*2"]:        
        x = self.softmax2d(x)
        
        expected_x = torch.sum(x * self.pos_x, dim=(-2, -1)) # [batch, channel]
        expected_y = torch.sum(x * self.pos_y, dim=(-2, -1)) # [batch, channel]
        
        feature_vector = torch.hstack([expected_x, expected_y]) # [batch, channel*2]
        
        return feature_vector