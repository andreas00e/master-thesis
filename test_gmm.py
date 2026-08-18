import logging 
import torch

from lightning.pytorch.loggers import WandbLogger
from models.discover.utils.cluster import KMeans, RGMM

import wandb
        
def main(): 
    run = wandb.init(project="scatter-plot")

    k = 5
    d = 4
    
    kmeans_kwargs = {
        "k": k, 
        "max_iter": 300, 
        "tol": 1e-4
    }
    
    rgmm_kwargs = {
        "k": k,  
        "d": d,
        "max_iter": 300, 
        "bml_weight": 0.1,
        "bml_alpha": 0.1, 
        "bml_beta": 0.2, 
        "sim": {
            "dim": 1, 
            "eps": 1.0e-6
        }
    } 
    
    device = "cuda" if torch.cuda.is_available() else "cpu" 
    
    kmeans = KMeans(**kmeans_kwargs).to(device)
    rgmm = RGMM(**rgmm_kwargs).to(device)
    
    lightning_logger = WandbLogger(project=run.project, id=run.id, resume="must")
    rgmm.logger = lightning_logger

    x = torch.rand(size=(10_000, d)).to(device)
    x_plus = torch.rand(size=(10_000, d)).to(device)
    
    weights, means, covs, labels = kmeans(x).values()
    
    _ = rgmm(x=x, x_plus=x_plus, weights=weights, means=means, covs=covs)
    print("FInished!")

if __name__ == "__main__": 
    main()