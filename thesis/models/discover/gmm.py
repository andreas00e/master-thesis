from typing import Dict, Optional 

import torch
import torch.nn as nn 
import torch.distributions as D
from torchtyping import TensorType
from torch.linalg import cholesky_ex, eigh

import lightning as pl 


class GMM(nn.Module): 
    def __init__(
        self, 
        k: int, # number of Gaussian distributions  
        d: int, # dimension of data and parameters of Gaussian distributions 
        alpha: float, 
        beta: float, 
        sim: Dict, 
        weights: Optional[TensorType["k", "d"]] = None, 
        means: Optional[TensorType["k", "d"]] = None, # mean vectors of Gaussian distributions 
        covs: Optional[TensorType["k", "d", "d"]] = None # covariance matrices of Gaussian distributions 
        ) -> None: 
        
        super().__init__()
        
        self.k = k 
        self.d = d
        self.alpha = alpha 
        self.beta = beta 
        self.sim = sim
        
        self.weights = weights if weights is not None else nn.Parameter(data=torch.ones(size = (self.k, )) / self.k)
        self.means = means if means is not None else nn.Parameter(data=torch.rand(size = (self.k, self.d)))
        self.covs = covs if covs is not None else nn.Parameter(data=torch.rand(size = (self.k, self.d, self.d)))
        self.tau = nn.Parameter(data=torch.ones(1, )) 
        self.lam = nn.Parameter(data=torch.ones(1, ))
        
        self.components, self.mixture = self.mixtureModel()
  
        self.sim = nn.CosineSimilarity(**self.sim)
        self.relu = nn.ReLU()

    def mixtureModel(self) -> torch.distributions.Distribution: 
        covs = self.covs

        if not self.is_pos_definite(covs.detach()):
            with torch.no_grad():
                covs = self.to_pos_definite(covs.detach())
                self.covs.copy_(covs)
            covs = self.covs

        components = D.MultivariateNormal(loc=self.means, covariance_matrix=covs)
        mixture = D.MixtureSameFamily(D.Categorical(self.weights), components)
        return components, mixture
    
    def is_pos_definite(self, A: TensorType["k", "d", "d"]) -> bool: 
        # ensure that the matrix is symmetrix first  
        if not torch.allclose(A, A.transpose(-1, -2)): 
            return False 
        
        # Cholesky decomposition
        _, info = cholesky_ex(A)
        return bool(torch.all(info == 0))
   
    def to_pos_definite(self, A: TensorType["k", "d", "d"]) -> TensorType["k", "d", "d"]: 
        # make all covariance matrices symmetric
        A = (A + A.transpose(-1, -2)) / 2
        
        # make all covariance matrices positive definite
        eigenvals, eigenvecs = eigh(A)
        eps = torch.abs(torch.amin(eigenvals, dim=-1, keepdim=True)) * 1.1 + 1e-6
        eigenvals = torch.where(eigenvals <= 0, eps, eigenvals)
        A = eigenvecs @ torch.diag_embed(eigenvals) @ eigenvecs.transpose(-1, -2)
        return A  

    def confidence(self, x: TensorType["n", "d"]) -> TensorType["n"]: 
        n = x.shape[0]
        probs = self.components.log_prob(x).exp() # [n, k] 
        vals = torch.topk(probs, k=2, dim=-1).values # [n, 2]
        
        rows = torch.arange(start=0, end=n, step=1) # [n]
        cols = torch.argmax(probs, dim=1) # [n]
        
        mask = torch.ones_like(probs, dtype=torch.bool) # [n, k] 
        mask[rows, cols] = False # [n, k] 
        probs = probs[mask].view(n, self.k-1) # [n, k-1]
        
        return torch.sigmoid(vals[:, 0]-vals[:, 1]) / torch.sigmoid(torch.var(probs, dim=-1)) # [n]
        
    def _posterior(self, x: TensorType["n", "d"]) -> TensorType["n", "k"]:
        log_weights = torch.log(self.weights)  # [k]
        log_probs = self.components.log_prob(x.unsqueeze(1))  # [n, k], unweighted log-density
        log_joint = log_probs + log_weights  # [n, k]: log(w_k * N(x|mu_k,Sigma_k))
        log_posterior = log_joint - torch.logsumexp(log_joint, dim=-1, keepdim=True)  # normalize over k
        return log_posterior.exp()  # [n, k]: true p_{i,k}
    
    def _get_m_s(self, p: TensorType["n", "k"]) -> TensorType["n", "2"]:
        probs_i, idxs_i = torch.topk(p, k=2, dim=-1) # [n, 2], [n, 2]: top 2 biggest posterior probabilities 
        probs_i_m = probs_i[:, 0] # [n, 1]: biggest posterior probabilities (p_max)
        probs_i_s = probs_i[:, 1] # [n, 1]: 2nd biggest posterior probabilities
        idxs_i_m = idxs_i[:, 0] # [n, 1]: indices of biggest posterior probabilities (p_max)
        idxs_i_s = idxs_i[:, 1] # [n, 1]: indices of 2nd biggest posterior probabilities
        return probs_i_m, probs_i_s, idxs_i_m, idxs_i_s
    
    def hard_negatives(self, x: TensorType["n", "d"], x_positive: TensorType["n", "d"]) -> TensorType["n", "d"]: 
        probs = self._posterior(x) # [n, k]: k posterior probabilites
        probs_i_m, probs_i_s, idxs_i_m, idxs_i_s = self._get_m_s(probs)
         
        x_negative = self.means[idxs_i_s, :] 
        
        cols_idxs = torch.argmax(probs, dim=1)[:, None] # [n] dimensions of the largest posterior probabilites       
        rows_idxs = torch.arange(0, probs.shape[0])[:, None] # [n]
        idxs = torch.concat(tensors=(rows_idxs, cols_idxs), dim=-1)
        ones = torch.ones_like(probs, dtype=torch.bool) # [n, k]
        ones[idxs] = False
        probs_j = probs[ones]
            
        conf_n = torch.sigmoid(probs_i_m - probs_i_s) # [n]
        conf_d = torch.var(probs_j, dim=-1) # [n]
        conf = conf_n / conf_d # [n]
        weight = probs_i_s / conf # [n]
        weight /= torch.sum(weight) # [n]
         
        d = torch.exp(self.sim(x, x_positive) / self.tau) + weight * torch.exp(self.sim(x, x_negative)) # [n]
        n = torch.exp(self.sim(x, x_positive) / self.tau) # [n]
        
        loss_cl = - torch.log(n / torch.sum(d)) # [1]        return loss_cl
        return loss_cl 
    
    def false_negatives(self, x: TensorType["n", "d"], x_positive: TensorType["n", "d"]) -> TensorType["n", "d"]:
        probs = self.posterior(x) # [n, k]: k posterior probabilites
        probs_i_s, probs_i_m, _, _ = self.posterior_m_s(probs)
        probs_i = torch.topk(probs, k=2, dim=-1).values # [n, 2]: top 2 biggest posterior probabilities 
        probs_i_m = probs_i[:, 0] # [n, 1]: biggest posterior probabilities (p_max)
        probs_i_s = probs_i[:, 1] # [n, 1]: 2nd biggest posterior probabilities
        
        idxs = torch.argmax(probs, dim=-1) # [n]: dimension of biggest posterior probabilities
        idxs = idxs.unsqueeze(1) == idxs.unsqueeze(0) # [n, n] 
        mask_idxs = ~torch.eye(x.shape[0], dtype=torch.bool) # [n, n]
        idxs &= mask_idxs # [n, n-1] # each rows is all x_j^m whose p_max is the same as x_i's p_max
        rows_idxs, cols_idxs = torch.nonzero(idxs, as_tuple=True) 

        probs_j_m = probs_i_m[cols_idxs, ...] # [..., 1]: x_j's biggest posterior probabilities
        probs_j_s = probs_i_s[cols_idxs, ...] # [..., 1] x_j's second biggest posterior probabilities
        probs_i_m = probs_i_m[rows_idxs, ...] # [..., 1]: x_i's (relevant) biggest posterior probabilities
        probs_i_s = probs_i_s[rows_idxs, ...] # [..., 1]: x_i's (relevant) second biggest posterior probabilities
        
        weight = torch.sigmoid(torch.abs(probs_i_s - probs_j_s)) * torch.sigmoid(torch.abs(probs_i_m - probs_j_m))
        _, inverse_indices = torch.unique(rows_idxs, return_inverse=True)
        counts = torch.zeros(size=(torch.unique(rows_idxs).numel(),))
        counts = torch.scatter_add(counts, 0, inverse_indices, torch.ones_like(weight))
        output_tensor = torch.zeros(size=(torch.unique(rows_idxs).numel(), ))
        weight_avg = torch.scatter_add(output_tensor, 0, inverse_indices, weight) 
        weight_avg /= counts
        weight_avg = weight_avg[inverse_indices, ...]
        
        x_j = x[cols_idxs, ...] 
        x = x[rows_idxs, ...]
        x_positive = x_positive[rows_idxs, ...]
        delta = self.sim(x, x_j) - self.sim(x, x_positive) 
        
        loss_bm = weight / weight_avg * (self.relu(delta + self.alpha) + (self.relu(-delta - self.beta)))
        return loss_bm
    
    def forward(self, x) -> float:
        x = x[:, 0, :]
        x_positive = x[:, 1, :]
        return self.hard_negatives(x, x_positive) + self.lam * self.false_negatives(x, x_positive)

def main():
    n = 100 
    d = 8
    
    gmm_kwargs = {
        "k": 20, 
        "d": 8, 
        "alpha": 0.1,
        "beta": 0.2, 
        "sim": {
            "dim": 1, 
            "eps": 0.8
        }
    } 
    
    x = torch.rand(size=(n, d))
    x_positive = torch.rand(size=(n, d))
    
    gmm = GMM(**gmm_kwargs)
    
    gmm.hard_negatives(x, x_positive)
    
    print("FINISHED!")

if __name__ == "__main__": 
    main() 