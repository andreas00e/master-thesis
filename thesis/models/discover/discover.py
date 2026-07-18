import lightning as pl 

from typing import Dict

from .visionBackbone import VisionBackbone
from .gmm import GMM 

class SkillDiscovery(pl.LightningModule): 
    def __init__(
        self, 
        vision_backbone: Dict, 
        gmm: Dict, 
        *args, **kwargs
        ) -> None:
        
        super().__init__()
        self.save_hyperparameters() 
        
        self.visionBackone = VisionBackbone(**vision_backbone)
        self.gmm = GMM(**gmm)        
        
    def forward(self, x ): 
        x = self.visionBackbone(x) 
        return x 
 
    def _shared_step(self, batch): 
        x, x_plus, _  = batch.values()
        x = self(x)
        x_plus = self(x_plus)
        return x, x_plus

    def train_step(self, batch, batch_idx): 
        x, x_plus = self._shared_step(self, batch)
        return x, x_plus
    
    def validation_step(self, batch, batch_idx):
        x, x_plus = self._shared_step(self, batch)
        return x, x_plus

    def test_step(self, batch, batch_idx): 
        x, x_plus = self._shared_step(self, batch)
        return x, x_plus