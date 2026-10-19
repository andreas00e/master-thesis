from matplotlib import pyplot as plt

import torch 
import torch.nn as nn 
from torchtyping import TensorType 


@torch.no_grad()
def plot_feature_maps(
    feature_tensor: TensorType["*"],
    num_plots: int=16, 
    title: str="Feature Maps"
    ) -> None:

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
    plt.show()


class ThreeLayerCNN(nn.Module):
    def __init__(
        self, 
        feature_dim: 64
        ) -> None:
        super().__init__()
        
        self.feature_dim = feature_dim
        
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.spatialSoftmax = SpatialSoftmax(height=10, width=10, channel=64)

        self.fc = nn.Sequential(
            nn.Linear(64 * 2, self.feature_dim),
            nn.ReLU(),
            nn.Linear(self.feature_dim, self.feature_dim)
        )

    def forward(self, x):
        out1 = self.pool1(self.relu1(self.conv1(x)))
        out2 = self.pool2(self.relu2(self.conv2(out1)))
        out3 = self.pool3(self.relu3(self.conv3(out2)))
        out4 = self.fc(self.spatialSoftmax(out3))
        
        return out1, out2, out3, out4
    
class WristVisionEncoder(nn.Module):
    def __init__(
        self, 
        n_channels: int=64, 
        feature_dim: int=64
        ) -> None:
        super().__init__()
        
        self.n_channels = n_channels
        self.feature_dim = feature_dim 
                
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2), # 32x42x42
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2), # 64x21x21
            
            nn.Conv2d(64, self.n_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2) # 64x10x10
        )
        
        self.spatialSoftmax = SpatialSoftmax(height=10, width=10, channel=64)

        self.fc = nn.Sequential(
            nn.Linear(64 * 2, self.feature_dim),
            nn.ReLU(),
            nn.Linear(self.feature_dim, self.feature_dim)
        )

    def forward(self, x: TensorType["batch", "channel", "height", "wdith"]) -> TensorType["batch", "feature_dim"]:
        x = self.cnn(x)
        x = self.spatialSoftmax(x)
        x = self.fc(x)              
        
        return x
    
class SpatialSoftmax(nn.Module):
    def __init__(
        self, 
        height: int,
        width: int, 
        channel: int
        ) -> None:
        super().__init__()
        
        self.height = height
        self.width = width
        self.channel = channel
        
        pos_x, pos_y = torch.meshgrid(
            torch.linspace(-1.0, 1.0, self.width),
            torch.linspace(-1.0, 1.0, self.height),
            indexing="ij"
        )
        
        self.register_buffer("pos_x", pos_x.reshape(1, 1, height, width))
        self.register_buffer("pos_y", pos_y.reshape(1, 1, height, width))
        
        self.softmax2d = nn.Softmax2d()

    def forward(self, x: TensorType["batch", "channel", "height", "width"]) -> TensorType["batch", "channel*2"]:        
        x = self.softmax2d(x)
        
        expected_x = torch.sum(x * self.pos_x, dim=(-2, -1)) # [batch, channel]
        expected_y = torch.sum(x * self.pos_y, dim=(-2, -1)) # [batch, channel]
        
        feature_vector = torch.hstack([expected_x, expected_y]) # [batch, channel*2]
        
        return feature_vector


def main(): 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    batch_size = 10 
    channel = 3 
    height = 84 
    width = 84

    input_tensor = torch.rand(size=(batch_size, channel, height, width), dtype=torch.float32, device=device)
    model = ThreeLayerCNN(feature_dim=64)
    model = model.to(device)
    
    out1, out2, out3, out4 = model(input_tensor)
    
    plot_feature_maps(out1)
    plot_feature_maps(out2)
    plot_feature_maps(out3)
    
    print(out4.shape)
    
    
if __name__ == "__main__": 
    main() 