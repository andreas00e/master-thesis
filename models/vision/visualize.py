import torch
from torchvision.models import resnet18
import lightning as pl 

class VisionEmbedder(pl.LightningModule): 
    def __init__(self):
        super().__init__()
        
        weights = torch.load(f="models/weights/resNet18_Weights_DEFAULT.pth")
        self.model = resnet18(weights=weights)
    
    def forward(self, x): 
        z_obs = self.vision_backbone(x) # [horizon, 1]
        z_obs = torch.mean(z_obs)
        return z_obs
    
    def test_step(self, batch, batch_idx): 
        rgb_obs, _ = batch.values()
        emb = self(rgb_obs)
        return emb