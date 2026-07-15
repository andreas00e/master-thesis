import random
from typing import Sequence

from torchvision.transforms import v2 


def get_transforms() -> Sequence[v2.transforms]: 

    transforms = v2.Compose(
        v2.ColorJitter(
            brigthness=random.uniform(0.2, 0.4), 
            contrast=random.uniform(0.2, 0.4), 
            saturation=random.uniform(0.2, 0.4), 
            hue=random.uniform(0.1, 0.2)
            ), 
        
        v2.GaussianBlur(
            kernek_size=random.choice([3, 5]),
            sigma=random.uniform(0.1, 2.0)
            ), 
        
        v2.RandomSolarize(
            threshold=0.5, 
            p=random.choice([0.2, 0.5]), 
            ), 
    ) 
    
    return transforms 