import torch

from models.discover.cluster import KMeans, GMM

        
def main(): 
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
    gmm = GMM(**gmm_kwargs).to(device)
    
    x = torch.rand(size=(100, d)).to(device)
    # x_plus = torch.rand(size=(100, d)).to(device)
    
    weights, means, covs, labels = kmeans(x).values()
    
    print(f"weights: {weights.shape}")
    print(f"means: {means.shape}")
    print(f"covs: {covs.shape}")
    
    exit()
    
    
    out = gmm(mode="update_gmm", x=x)
    # out = gmm(mode="update_encoder", x=x, x_plus=x_plus)
    print("Hello")

if __name__ == "__main__": 
    main()