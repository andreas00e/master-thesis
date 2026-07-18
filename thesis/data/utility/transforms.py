import random
from typing import Callable, List, Sequence

from torchvision.transforms import v2 
import torch.nn as nn 

TRANSFORMS = {
    "grayscale": 
        v2.RandomGrayscale(p=random.uniform(0, 0.5)), 
    
    "gaussian_blur": 
        v2.GaussianBlur(
            kernel_size=random.choice([3, 5]),
            sigma=random.uniform(0.1, 2.0)
            ), 
        
    "color_jitter": 
        v2.ColorJitter(
            brightness=random.uniform(0.2, 0.4), 
            contrast=random.uniform(0.2, 0.4), 
            saturation=random.uniform(0.2, 0.4), 
            hue=random.uniform(0.1, 0.2)
            ), 
    
    "solarize": 
        v2.RandomSolarize(
            threshold=0.5, 
            p=random.choice([0.2, 0.5]), 
            ),     
        
    "rotate": 
        v2.RandomRotation((-45, 45))
}

def get_transforms(transforms: List[str]) -> Sequence[Callable]: 
    transforms = [TRANSFORMS[k] for k in transforms]
    transforms =[*nn.Sequential(*transforms)]
    
    return transforms