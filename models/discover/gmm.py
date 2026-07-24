from typing import Dict, Optional 

import torch
import torch.nn as nn 
import torch.distributions as D
from torchtyping import TensorType
from torch.linalg import cholesky_ex, eigh


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
        
        self.weights = weights if isinstance(weights, nn.Parameter) else nn.Parameter(weights) if weights is not None else nn.Parameter(data=torch.ones(size=(self.k, )) / self.k)
        self.means = means if isinstance(means, nn.Parameter) else nn.Parameter(means) if means is not None else nn.Parameter(data=torch.rand(size=(self.k, self.d)))
        self.covs = covs if isinstance(covs, nn.Parameter) else nn.Parameter(covs) if covs is not None else nn.Parameter(data=torch.rand(size=(self.k, self.d, self.d)))
        self.tau = nn.Parameter(data=torch.ones(1, )) 
        self.lam = nn.Parameter(data=torch.ones(1, ))
        
        self.components = None
  
        self.sim = nn.CosineSimilarity(**self.sim)
        self.relu = nn.ReLU()

    def mixtureModel(self) -> torch.distributions.Distribution: 

        if not self.is_pos_definite(self.covs.data): # we don't have to change the Parameter if we only take the data here
            self.covs.data = self.to_pos_definite(self.covs.data)
        
        components = D.MultivariateNormal(self.means, self.covs)
        # mixture = D.MixtureSameFamily(D.Categorical(self.weights), components)
        return components
    
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
    
    def _hard_negatives(self, x: TensorType["n", "d"], x_positive: TensorType["n", "d"]) -> TensorType["n", "d"]: 
        probs = self._posterior(x) # [batch, k]: k posterior probabilites
        probs_i_m, probs_i_s, _, idxs_i_s = self._get_m_s(probs)
         
        x_negative = self.means[idxs_i_s, :] 
        
        cols_idxs = torch.argmax(probs, dim=1)[:, None] # [batch, 1] dimensions of the largest posterior probabilites       
        rows_idxs = torch.arange(0, probs.shape[0], device=x.device)[:, None] # [batch, 1]
        mask = torch.ones_like(probs, dtype=torch.bool, device=x.device) # [n, k]
        mask[rows_idxs, cols_idxs] = False
        probs_j = probs[mask] # set every biggest row value to zero
            
        conf_n = torch.sigmoid(probs_i_m - probs_i_s) # [n]
        conf_d = torch.var(probs_j, dim=-1) # [n]
        conf = conf_n / conf_d # [n]
        weight = probs_i_s / conf # [n]
        weight /= torch.sum(weight) # [n]
         
        d = torch.exp(self.sim(x, x_positive) / self.tau) + weight * torch.exp(self.sim(x, x_negative)) # [n]
        n = torch.exp(self.sim(x, x_positive) / self.tau) # [n]
        
        loss_cl = - torch.log(n / torch.sum(d)) # [1] 
        
        return loss_cl 
    
    def _false_negatives(self, x: TensorType["n", "d"], x_positive: TensorType["n", "d"]) -> TensorType["n", "d"]:
        n = x.shape[0]
        
        probs = self._posterior(x) # [n, k]: k posterior probabilites
        probs_i_s, probs_i_m, _, _ = self._get_m_s(probs)
        probs_i = torch.topk(probs, k=2, dim=-1).values # [n, 2]: top 2 biggest posterior probabilities 
        probs_i_m = probs_i[:, 0] # [n, 1]: biggest posterior probabilities (p_max)
        probs_i_s = probs_i[:, 1] # [n, 1]: 2nd biggest posterior probabilities
        
        idxs = torch.argmax(probs, dim=-1) # [n]: dimension of biggest posterior probabilities
        idxs = idxs.unsqueeze(1) == idxs.unsqueeze(0) # [n, n] 
        mask_idxs = ~torch.eye(n, dtype=torch.bool, device=x.device) # [n, n]
        idxs &= mask_idxs # [n, n-1] # each rows is all x_j^m whose p_max is the same as x_i's p_max
        rows_idxs, cols_idxs = torch.nonzero(idxs, as_tuple=True) 

        probs_j_m = probs_i_m[cols_idxs, ...] # [..., 1]: x_j's biggest posterior probabilities
        probs_j_s = probs_i_s[cols_idxs, ...] # [..., 1] x_j's second biggest posterior probabilities
        probs_i_m = probs_i_m[rows_idxs, ...] # [..., 1]: x_i's (relevant) biggest posterior probabilities
        probs_i_s = probs_i_s[rows_idxs, ...] # [..., 1]: x_i's (relevant) second biggest posterior probabilities
        
        weight = torch.sigmoid(torch.abs(probs_i_s - probs_j_s)) * torch.sigmoid(torch.abs(probs_i_m - probs_j_m))
        unique_elements, inverse_indices = torch.unique(rows_idxs, return_inverse=True)
        counts = torch.zeros(size=(torch.unique(rows_idxs).numel(),), device=x.device)
        counts = torch.scatter_add(counts, 0, inverse_indices, torch.ones_like(weight, device=x.device))
        output_tensor = torch.zeros(size=(torch.unique(rows_idxs).numel(), ), device=x.device)
        weight_avg = torch.scatter_add(output_tensor, 0, inverse_indices, weight) 
        weight_avg /= counts
        weight_avg = weight_avg[inverse_indices, ...]
        
        x_j = x[cols_idxs, ...] 
        x = x[rows_idxs, ...]
        x_positive = x_positive[rows_idxs, ...]
        delta = self.sim(x, x_j) - self.sim(x, x_positive) 
        
        loss_bml = weight / weight_avg * (self.relu(delta + self.alpha) + (self.relu(-delta - self.beta)))
        loss_bml = torch.scatter_add(output_tensor, 0, inverse_indices, loss_bml)
        
        if unique_elements.shape != n: 
            out = torch.zeros(size=(n, ), device=x.device)
            out[unique_elements] = loss_bml
            loss_bml = out 

        return loss_bml
    
    def _entropy(self, x: TensorType["*"]) -> None: 
        probs = self._posterior(x) # [n, k]
        idxs = torch.argmax(probs, dim=-1) # [n, k]
        
        probs = probs[torch.arange(100, device=x.device), idxs] # [n, k]
        probs = probs * torch.log(probs) # [n, k]
        
        unique_elements, inverse_indices = torch.unique(idxs, return_inverse=True)
        output_tensor = torch.zeros(size=(unique_elements.shape[0], ), device=x.device)
        
        E_k = torch.zeros(size=(self.k, ), device=x.device)
        E_k[unique_elements] = - torch.scatter_add(output_tensor, 0, inverse_indices, probs)
        
        return E_k 
    
    def _zero_posterior_update(self, x: TensorType["*"]) -> None:
        probs = self._posterior(x) # [n, k]
        weights = self.weights[None, ...].repeat(x.shape[0], 1) # [k] -> [n, k]
        
        num = probs * weights # [n, k]
        denom = torch.sum(num, dim=1, keepdim=True).repeat(1, weights.shape[1]) # [n, k]
        probs = num / denom 
        
        return probs
        
    def _first_posterior_update(self, x): 
        E_k = self._entropy(x) # [k]
        
        G_min = torch.argmin(E_k) # [1]: distribution with the largest entropy
        G_max = torch.argmax(E_k) # [1]: distribution with the smallest entropy

        probs = self._posterior(x) # [n, k]: k posterior probabilities
        idxs = torch.argmax(probs, dim=-1) # [n]
        
        idxs_min = torch.nonzero(G_min == idxs) 
        idxs_max = torch.nonzero(G_max == idxs) 
        
        count_G_min = idxs_min.shape[0]
        count_G_max = idxs_max.shape[0]
        
        x_min = x[idxs_min] # [count_G_min, self.d]
        x_max = x[idxs_max] # [count_G_max, self.d]
        
        num_means = torch.sum(torch.concat(tensors=(x_min, x_max), dim=0), dim=0)
        means = num_means / (count_G_max + count_G_min)       

        weights =  (count_G_max + count_G_min) / (x.shape[0] - 1)

        num_covs = (x - means)[:, :, None] * (x - means)[:, None, :] # [n, d, 1] * [n, 1, d] -> [n, d, d]
        num_covs = torch.sum(num_covs, dim=0)  
        covs = num_covs / (count_G_max + count_G_min)
        
        return weights, means, covs 

    def _second_posterior_update(self, x): 
        E_k = self._entropy(x) # [k]
        
        G_min = torch.argmin(E_k) # [1]: distribution with the largest entropy
        G_max = torch.argmax(E_k) # [1]: distribution with the smallest entropy

        probs = self._posterior(x) # [n, k]: k posterior probabilities
        idxs = torch.argmax(probs, dim=-1) # [n]
        
        idxs_min = torch.nonzero(G_min == idxs) 
        idxs_max = torch.nonzero(G_max == idxs) 
        
        count_G_max = idxs_max.shape[0]
        
        x_min = x[idxs_min]
        
        weights = count_G_max / x.shape[0]
        
        means = torch.sum(x_min, dim=0) / count_G_max
        
        num_covs = (x - means)[:, :, None] * (x - means)[:, None, :] # [n, d, 1] * [n, 1, d] -> [n, d, d]
        num_covs = torch.sum(num_covs, dim=0) 
        covs = num_covs / count_G_max  
        
        return weights, means, covs 
    
    def forward(self, x, x_plus) -> TensorType["*"]:
        self.components = self.mixtureModel()
        
        loss_cl = self._hard_negatives(x, x_plus) 
        loss_bml = self._false_negatives(x, x_plus)
        
        return loss_cl + self.lam * loss_bml 