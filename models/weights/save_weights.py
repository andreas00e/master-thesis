import os
import torch 

from r3m import load_r3m


def main(): 
    version = "resnet50" # resnet18, resnet34, resnet50

    
    model = load_r3m(version)

    torch.save(model.state_dict(), f"r3m_{version}_weights.pth")
    
if __name__ == "__main__": 
    main()

