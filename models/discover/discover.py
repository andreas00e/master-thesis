import wandb
from typing import Dict

import lightning as pl 

from .gmm import GMM 
from .visionBackbone import VisionBackbone
from .visionEncoder import VisionEncoder


class SkillDiscovery(pl.LightningModule): 
    def __init__(
        self, 
        vision_backbone_kwargs: Dict, 
        vision_encoder_kwargs: Dict, 
        gmm_kwargs: Dict
        ) -> None:
        
        super().__init__()
        self.save_hyperparameters() 
                
        self.visionBackbone = VisionBackbone(**vision_backbone_kwargs)
        self.visionEncoder = VisionEncoder(**vision_encoder_kwargs)
        self.gmm = GMM(**gmm_kwargs)      
        
    def forward(self, x): 
        x = self.visionBackbone(x) 
        x = self.visionEncoder(x)
        return x 
 
    def _shared_step(self, batch): 
        x, x_plus, _ = batch.values()
        
        if self.logger and hasattr(self.logger, "log_image"): 
            self.logger.log_image(
                key = "x", 
                image = [wandb.Image(img) for img in x.detach().cpu().numpy()]
            )
            self.logger.log_image(
                key = "x_plus", 
                image = [wandb.Image(img) for img in x_plus.detach().cpu().numpy()]
            )
        
        x_out = self(x)
        x_plus_out = self(x_plus)
        
        return x_out, x_plus_out

    def train_step(self, batch, batch_idx): 
        x, x_plus = self._shared_step(batch)
        
        return x, x_plus
    
    def validation_step(self, batch, batch_idx):
        x, x_plus = self._shared_step(batch)
        
        return x, x_plus

    def test_step(self, batch, batch_idx): 
        x, x_plus = self._shared_step(batch)
        
        return x, x_plus