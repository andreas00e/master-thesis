import torch


from models.discover.utils.cluster import KMeans, RGMM

import wandb
        
def main(): 
    wandb.init(project="scatter-plot")
    
    
    k = 20
    d = 4
    
    kmeans_kwargs = {
        "k": k, 
        "max_iter": 300, 
        "tol": 1e-4
    }
    
    gmm_kwargs = {
        "k": k,  
        "d": d,
        "alpha": 0.1, 
        "beta": 0.2, 
        "sim": {
            "dim": 1, 
            "eps": 1.0e-6
        }
    } 
    
    device = "cuda" if torch.cuda.is_available() else "cpu"   
    
    kmeans = KMeans(**kmeans_kwargs).to(device)

    x = torch.rand(size=(10_000, d)).to(device)
    # x_plus = torch.rand(size=(100, d)).to(device)
    
    weights, means, covs, labels = kmeans(x).values()
    
    # columns = [f"dim_{i}" for i in range(d)] + ["label"]
    # data = [list(vec)+[lbl] for vec, lbl in zip(x, labels)]
    
    # table = wandb.Table(columns=columns, data=data)
    # wandb.log({"embeddings": table})
    
    
    gmm = RGMM(weights=weights, means=means, covs=covs, **gmm_kwargs).to(device)
    _ = gmm(mode="update_gmm", x=x)
    # out = gmm(mode="update_encoder", x=x, x_plus=x_plus)
    print("Hello")

if __name__ == "__main__": 
    main()