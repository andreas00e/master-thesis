import torch
from typing import List
from torchvision.transforms import v2


TRANSFORMS = {  
    "to_image": 
        lambda: v2.ToImage(),

    "resize": 
        lambda: v2.Resize(
            size=(224, 224), 
            antialisas=True
        ),
    
    "gaussian_blur": 
        lambda: v2.GaussianBlur(
            kernel_size=5,
            sigma=(0.1, 2.0)
        ), 
        
    "color_jitter": 
        lambda: v2.ColorJitter(
            brightness=(0.2, 0.4), 
            contrast=(0.2, 0.4), 
            saturation=(0.2, 0.4), 
            hue=(0.1, 0.2)
        ), 

    "rotate": 
        lambda: v2.RandomRotation((-45, 45)), 

    "to_dtype": 
        lambda: v2.ToDtype(torch.float32, scale=True),
    
    "solarize": 
        lambda: v2.RandomSolarize(
            threshold=0.5,             
            p=0.5 
        ),   

    "normalize":
        lambda: v2.Normalize(
            mean=[0.485, 0.456, 0.406], #  ImageNet means
            std=[0.229, 0.224, 0.225] # ImageNet standard deviations
        )
}

def get_transforms(transforms: List[str]) -> v2.Compose: 
    return v2.Compose([TRANSFORMS[k]() for k in transforms])