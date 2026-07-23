import random
from typing import Callable, List, Sequence

import torch 
from torchvision.transforms import v2 


TRANSFORMS = {  
    "grayscale": 
        lambda: v2.RandomGrayscale(p = random.uniform(0, 0.5)), 
    
    "gaussian_blur": 
        lambda: v2.GaussianBlur(
            kernel_size = random.choice([3, 5]),
            sigma = random.uniform(0.1, 2.0)
            ), 
        
    "color_jitter": 
        lambda: v2.ColorJitter(
            brightness = random.uniform(0.2, 0.4), 
            contrast = random.uniform(0.2, 0.4), 
            saturation = random.uniform(0.2, 0.4), 
            hue = random.uniform(0.1, 0.2)
            ), 
    
    "solarize": 
        lambda: v2.RandomSolarize(
            threshold = 0.5, 
            p = random.choice([0.2, 0.5]), 
            ),     
        
    "rotate": 
        lambda: v2.RandomRotation((-45, 45)), 
        
    "to_image": 
        lambda: v2.ToImage(),
    
    "to_dtype": 
        lambda: v2.ToDtype(torch.float32, scale=True)
}

def get_transforms(transforms: List[str]) -> Sequence[Callable]: 
    transforms = v2.Compose([TRANSFORMS[k]() for k in transforms])
    
    return transforms