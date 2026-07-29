import torch
import torch.nn as nn
import torch.nn.functional as F

from torchtyping import TensorType

class KMeans(nn.Module):
    def __init__(self, n_clusters: int=3, max_iter: int=300, tol: float=1e-4):
        super().__init__()
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.centroids = nn.Parameter(torch.empty(0), requires_grad=False) # [n_cluster, d]
        self.is_fitted = False
         
    def _compute_distances(self, X: TensorType["n", "d"]) -> TensorType["n"]:
        # ||X - C||^2 = ||X||^2 + ||C||^2 - 2*X*C.T  
        x_norm = (X ** 2).sum(dim=1, keepdim=True)
        c_norm = (self.centroids ** 2).sum(dim=1, keepdim=True).T
        norm = x_norm + c_norm  
        distances = torch.addmm(norm, X, self.centroids.T, alpha=-2, beta=1)
        return distances

    def forward(self, X: TensorType["n", "d"]) -> TensorType["n"]:
        if not self.is_fitted:
            _ = self.fit(X)
            
        distances = self._compute_distances(X)
        return torch.argmin(distances, dim=1)

    def fit(self, X: TensorType["n", "d"]) -> bool:
        n, d = X.shape # [n, d]

        random_indices = torch.randperm(n, device=X.device)[:self.n_clusters] # initialize centroids from a random permutation of the data points
        self.centroids.data = X[random_indices].clone()  

        for _ in range(self.max_iter):
            old_centroids = self.centroids.data.clone() #

            distances = self._compute_distances(X) # [n, n_clusters]:

            cluster_assignments = torch.argmin(distances, dim=1) # [n]: hard cluster assignment

            one_hot = F.one_hot(cluster_assignments, num_classes=self.n_clusters).to(X.dtype) # [n, k]: one-hot centroid mapping matrix 
                      
            counts = torch.clamp(torch.sum(one_hot, dim=0, keepdim=True).T, min=1e-8) # [k, 1] # count points in each cluster (prevent division by zero with small epsilon)

            self.centroids.data = (one_hot.T @ X) / counts # [k, n] @ [n, d] / [k, 1]

            center_shift = torch.norm(self.centroids.data - old_centroids)
            if center_shift < self.tol: # check for convergence by tracking the shift of the centroids
                break

        self.is_fitted = True
        return self

if __name__ == "__main__":
    kmeans_layer = KMeans(n_clusters=4)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kmeans_layer.to(device)
    
    train_data = torch.randn(50000, 64, device=device)
    test_batch = torch.randn(128, 64, device=device)

    with torch.no_grad():
        labels = kmeans_layer(test_batch) # Triggers the forward method
        
    print(f"Batch prediction labels shape: {labels.shape}")
    print(f"Module parameter shape: {kmeans_layer.centroids.shape}")