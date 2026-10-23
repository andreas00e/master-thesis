import torch
import torch.nn as nn 
from typing import List
from torchvision.transforms import v2


TRANSFORMS = {  
    "to_image": 
        lambda _: v2.ToImage(),
    
    "erasing": 
        lambda x: v2.RandomErasing(
            p=min(0.2*x, 1.0),
            scale=(0.05, 0.15)            
            ), 
        
    "rotate": 
        lambda x: v2.RandomRotation(
            degrees=(max(-30, -10*x), min(10*x, 30))
        ), 
    
    "color_jitter": 
        lambda x: v2.RandomApply(
            [
            v2.ColorJitter(
            brightness=0.4,
            contrast=0.4, 
            saturation=0.1,  
            hue=0.1)
            ], 
            p=min(0.8*x, 1)
        ),  

    "solarize": 
        lambda x: v2.RandomSolarize(
            threshold=0.5,             
            p=min(0.2*x, 1.0)
        ),  

    "gaussian_blur": 
        lambda _: v2.RandomApply(
            [
            v2.GaussianBlur(
            kernel_size=5,
            sigma=(0.1, 2.0))
            ]
        ),  

    "grayscale": 
        lambda x: v2.RandomGrayscale(
            p=min(0.2*x, 1.0)
            ), 
  
    "to_dtype_cnn": 
        lambda _: v2.ToDtype(
            dtype=torch.float32, 
            scale=True
        ),
    
    "to_dtype_r3m": 
        lambda _: v2.ToDtype(
            dtype=torch.float32, 
            scale=False, 
        ),

    "resize_cnn": 
        lambda _: v2.Resize(
            # size=(112, 112)
            size=(84, 84)
        ),
        
    "resize_crop_cnn":
        lambda _: v2.RandomResizedCrop(
            size=(112, 112), 
            scale=(0.7, 1.0), 
            ratio=(0.9, 1.1), 
            antialias=True, 
         ),
    
    "resize_r3m": 
        lambda _: v2.Resize(
            size=(224, 224)
        ),

    "normalize":
        lambda _: v2.Normalize(
            mean=[0.485, 0.456, 0.406], # ImageNet means
            std=[0.229, 0.224, 0.225]   # ImageNet standard deviations
        )
}

def get_transforms(transforms: List[str], x: int=1) -> nn.Sequential:  
    x = x / (x + 1)
    
    return nn.Sequential(*[TRANSFORMS[k](x) for k in transforms])