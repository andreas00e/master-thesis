from typing import Dict, Optional 

import torch
import torch.nn as nn 
import torch.distributions as D
from torchtyping import TensorType
from torch.linalg import cholesky, cholesky_ex, eigh

import lightning as pl 


class GMM(): 
    def __init__(
        self, 
        k: int, # number of Gaussian distributions  
        d: int, # dimension of data and parameters of Gaussian distributions 
        alpha: float, 
        beta: float, 
        cos: Dict, 
        weights: Optional[TensorType["k", "d"]] = None, 
        means: Optional[TensorType["k", "d"]] = None, # mean vectors of Gaussian distributions 
        covs: Optional[TensorType["k", "d", "d"]] = None # covariance matrices of Gaussian distributions 
        ) -> None: 
        
        super().__init__()
        
        self.k = k 
        self.d = d
        self.alpha = alpha 
        self.beta = beta 
        self.cos = cos
        
        self.weights = weights if weights is not None else torch.ones(size=(self.k, )) / self.k
        self.means = means if means is not None else torch.rand(size=(self.k, self.d))
        self.covs = covs if covs is not None else torch.rand(size=(self.k, self.d, self.d))
        
        self.components, self.mixture = self.mixtureModel()
  
        self.cos = nn.CosineSimilarity(**self.cos)
        self.relu = nn.ReLU()

    
    def mixtureModel(self) -> torch.distributions.Distribution: 
        if not self.is_pos_definite(self.covs): 
            self.covs = self.to_pos_definite(self.covs)
          
        components = D.MultivariateNormal(
            loc=self.means, 
            scale_tril=cholesky(self.covs)
        )
        
        mixture = D.MixtureSameFamily(
            D.Categorical(self.weights),
            components
        )
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

    @property
    def mean_components(self):
        return self.components.mean
       
    def hard_negatives(self, x: TensorType["n", "d"]) -> TensorType["n", "d"]: 
        probs = self.components.log_prob(x).exp() 
        _, idcs  = torch.topk(probs, k=2, dim=-1)
        x_minus = self.components.mean[idcs[:, 1], :] 
        return x_minus
    
    def false_negatives(self, x: TensorType["n", "d"], x_plus: TensorType["n", "d"]) -> TensorType["n", "d"]:
        probs = self.components.log_prob(x.unsqueeze(1)).exp() # [n, 1, k]: k posterior probabilites
        
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
                
        x_j = x[rows_idxs, ...]
        x_plus = x_plus[rows_idxs,...]
        x = x[rows_idxs, ...] 
        delta = self.cos(x, x_j) - self.cos(x, x_plus)
        
        loss_bm = weight * (self.relu(delta + self.alpha) + (self.relu(-delta - self.beta)))
        return loss_bm 