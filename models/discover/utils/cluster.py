import numpy as np 
from typing import Dict, Optional, Tuple 

import torch
import torch.nn as nn 
import torch.nn.functional as F
import torch.distributions as D
from torchtyping import TensorType
from torch.linalg import cholesky_ex, eigh

import wandb 


class KMeans(nn.Module):
    def __init__(
        self, 
        k: int=3, 
        max_iter: Optional[int]=300, 
        tol: Optional[float]=1e-4
        ) -> None:
        super().__init__()
        
        self.k = k
        self.max_iter = max_iter
        self.tol = tol
        
        weights = torch.empty(size=(1, 1), dtype=torch.float32)
        means = torch.empty(size=(1, 1), dtype=torch.float32)
        covs = torch.empty(size=(1, 1, 1), dtype=torch.float32)
        
        self.register_buffer("weights", weights) # [k, 1]
        self.register_buffer("means", means) # [k, d]
        self.register_buffer("covs", covs) # [k, d, d]
        
        self.is_fitted = False
         
    def _compute_distances(self, x: TensorType["n", "d"]) -> TensorType["n", "k"]:
        # ||X - C||^2 = ||X||^2 + ||C||^2 - 2*X*C.T  
        x_norm = (x ** 2).sum(dim=1, keepdim=True) # [n, 1]
        c_norm = (self.means ** 2).sum(dim=1, keepdim=True).T # [1, k]
        distances = x_norm + c_norm - 2 * x @ self.means.T # [n, k]
        return distances
     
    def _fit(self, x: TensorType["n", "d"]) -> "KMeans":
        n, d = x.shape
        assert n >= self.k, "Need at least k points"
        
        # Random init from data 
        rand_idxs = torch.randperm(n, device=x.device)[:self.k]
        self.means = self.means.expand(self.k, -1) # XXX: Make sure that that has not already been done before 
        self.means.data = x.clone()[rand_idxs]
                
        for _ in range(self.max_iter):
            old_means = self.means.data.clone() 
            distances = self._compute_distances(x) # [n, k]:
            labels = torch.argmin(distances, dim=1) # [n]: hard cluster assignments
            one_hot = F.one_hot(labels, num_classes=self.k).to(x.dtype) # [n, k]: one-hot centroid mapping matrix (one in column of assigned cluster; zero elsewhere)      
            counts = torch.clamp(torch.sum(one_hot, dim=0, keepdim=True).T, min=1e-8) # [k, 1] # count points in each cluster (prevent division by zero with small epsilon)

            self.means.copy_((one_hot.T @ x) / counts) # [k, n] @ [n, d] / [k, 1]
            # Check for convergence by tracking the shift of the centroids
            if torch.norm(self.means - old_means) < self.tol:
                break

        self.is_fitted = True
        return self
    
    def _get_covariance(self, x: TensorType["n", "d"], labels: TensorType["n"]) -> TensorType["k", "d", "d"]: 
        covs = []
        
        for k in range(self.k):
            mask = labels == k
            data = x[mask]
            
            if mask.sum().item() <= 1: # not enough samples to estiamte covariance matrix 
                cov = torch.eye(x.shape[-1], dtype=x.dtype, device=x.device) # [d, d]
            else:
                cov = torch.cov(data.T) # [d, d]
            covs.append(cov)
            
        covs = torch.stack(covs, dim=0) # [k, d, d]
        return covs 
    
    def forward(self, x: TensorType["n", "d"]) -> Dict[str,TensorType["*"]]:
        if not self.is_fitted:
            self._fit(x)
            
        distances = self._compute_distances(x)
        labels = torch.argmin(distances, dim=1) # [n]
        counts = torch.zeros(self.k, dtype=torch.long, device=x.device)
        counts.scatter_add_(0, labels, torch.ones_like(labels))

        self.weights = counts.float() / x.shape[0] # [k]: sum to 1 
        
        self.covs = self._get_covariance(x, labels) # [k, d, d]
        
        return {
            "weights": self.weights, # [k]
            "means": self.means.data, # [k, d]
            "covs": self.covs, # [k, d, d]
            "labels": labels # [n]
        } 

class MBKMeans(nn.Module): 
    def __init__(
        self, 
        k: int, 
        max_iter: int,
        ) -> None: 
        
        self.k = k # number of clusters 
        self.max_iter = max_iter # number of maximum iterations 
       
        self.centers = None
        self.counts = None
        
    def _init_cc(self, x: TensorType["b", "n", "d"]): 
        idxs = torch.randperm(x.shape[0])[:self.k]
        centers = x[idxs] # [k, n, d]
        
        return centers 

    def forward(self, x: TensorType["mb", "n", "d"]) -> None: # mb: mini-batch
        x = x.view(-1, delattr) # [mb*n, d]
        
        distances = self._compute_distances(x) # [mb*n, k]
        labels = torch.argmin(distances, dim=1) # [mb*n]: hard cluster assignments
        one_hot = F.one_hot(labels, num_classes=self.k).to(x.dtype) # [mb*n, k]: one-hot centroid mapping matrix       
        self.counts = torch.clamp(torch.sum(one_hot, dim=0, keepdim=True).T, min=1e-8) # [k, 1] # per-center counts (prevent division by zero with small epsilon)
        
        lr = 1 / self.counts # [k, 1]
        self.centers = (1 - lr) * self.centers + lr * x 
        
    def _compute_distances(self, x: TensorType["mb*n", "d"]) -> TensorType["mb*n", "k"]:
        # ||X - C||^2 = ||X||^2 + ||C||^2 - 2*X*C.T  
        x_norm = (x ** 2).sum(dim=1, keepdim=True) # [n, 1]
        c_norm = (self.centers ** 2).sum(dim=1, keepdim=True).T # [1, k]
        distances = x_norm + c_norm - 2 * x @ self.centers.T # [n, k]
        return distances

class RGMM(nn.Module): 
    def __init__(
        self, 
        k: int, # number of Gaussian distributions  
        d: int, # dimension of data and parameters of Gaussian distributions 
        max_iter: int,
        bml_weight: float, 
        bml_alpha: float, 
        bml_beta: float, 
        sim: Dict, # parameters of cosine similarity
        weights: Optional[TensorType["k", "d"]] = None, # mixture weights of GMM's individual distributions 
        means: Optional[TensorType["k", "d"]] = None, # mean vectors of Gaussian distributions 
        covs: Optional[TensorType["k", "d", "d"]] = None # covariance matrices of Gaussian distributions 
        ) -> None: 
        super().__init__()
        
        self.k = k 
        self.d = d
        self.max_iter = max_iter
        self.bml_weight = bml_weight
        self.bml_alpha = bml_alpha 
        self.bml_beta = bml_beta 
        self.sim = sim
        
        self.weights = weights if isinstance(weights, nn.Parameter) else nn.Parameter(weights) if weights is not None else nn.Parameter(data=torch.ones(size=(self.k, )) / self.k) # [k]
        self.means = means if isinstance(means, nn.Parameter) else nn.Parameter(means) if means is not None else nn.Parameter(data=torch.randn(size=(self.k, self.d))) # [k, d]
        self.covs = covs if isinstance(covs, nn.Parameter) else nn.Parameter(covs) if covs is not None else nn.Parameter(data=torch.eye(self.d).unsqueeze(0).repeat(self.k, 1, 1)) # [k, d, d]
        self.tau = nn.Parameter(data=torch.ones((1, ), dtype=torch.float64)) # [1]
        self.lam = nn.Parameter(data=torch.ones((1, ), dtype=torch.float64)) # [1]
        
        self.component_distribution, self.dtype, self.device = [None]*3
        self.G_min, self.G_max, self.count_G_min, self.count_G_max, self.idxs_min, self.idxs_max = [0]*6 
  
        self.sim = nn.CosineSimilarity(**self.sim)
        self.relu = nn.ReLU()
        
    def _get_mixture_model(self, means: TensorType["k", "d"], covs: TensorType["k", "d", "d"]) -> D.MultivariateNormal:        
        if not self._is_pos_def(covs):
            covs = self._to_pos_def(covs)
        
        component_distribution = D.MultivariateNormal(loc=means, covariance_matrix=covs)
        
        return component_distribution
    
    def _is_pos_def(self, A: TensorType["k", "d", "d"]) -> bool:
        if not torch.allclose(A, A.transpose(-1, -2)): # ensure that the matrix is symmetrix first  
            return False 
        
        _, info = cholesky_ex(A) # Cholesky decomposition

        return bool(torch.all(info == 0))
    
    def _to_pos_def(self, A: TensorType["k", "d", "d"]) -> TensorType["k", "d", "d"]:
        A = (A + A.transpose(-1, -2)) / 2  # make all covariance matrices symmetric

        # Make all covariance matrices positive definite
        eigenvals, _ = eigh(A)
        min_eig = eigenvals.amin(dim=-1)  

        shift = torch.clamp(-min_eig, min=0) * 1.1 + 1e-6  # TODO: MOVE THE HYPERPARAMETERS TO THE CONFIG
        eye = torch.eye(A.shape[-1], device=A.device, dtype=A.dtype)
        A = A + shift.view(-1, 1, 1) * eye

        return A

    def _posterior(
        self,
        x: TensorType["n", "d"],
        weights: TensorType["k"], 
        distribution: D.MultivariateNormal
        
    ) -> TensorType["n", "k"]:
        log_probs = distribution.log_prob(x.unsqueeze(1)) # [n, k]: unweighted log-density
        log_weights = torch.log(weights) # [k]
        log_joint = log_probs + log_weights # [n, k]: log(w_k * log(N(x | mu_k, Sigma_k)) = log(w_k) + log(N(x | mu_k, Sigma_k))
        log_posterior = log_joint - torch.logsumexp(log_joint, dim=-1, keepdim=True) # [n, k]: normalize over k 
        
        return log_posterior.exp() # [n, k]: posterior probabilities p_{i,k} 
    
    def _confidence(self, x: TensorType["n", "d"]) -> TensorType["n"]: 
        n = x.shape[0]
        probs = self.components.log_prob(x).exp() # [b, n, k] 
        vals = torch.topk(probs, k=2, dim=-1).values # [b, n, 2]
        
        rows = torch.arange(start=0, end=n, step=1) # [b, n]
        cols = torch.argmax(probs, dim=1) # [b, n]
        
        mask = torch.ones_like(probs, dtype=torch.bool) # [b, n, k] 
        mask[rows, cols] = False # [b, n, k] 
        probs = probs[mask].view(n, self.k-1) # [b, n, k-1]
        
        return torch.sigmoid(vals[:, 0]-vals[:, 1]) / torch.sigmoid(torch.var(probs, dim=-1)) # [n]
    
    def _get_m_s(self, p: TensorType["n", "k"]) -> TensorType["n", "2"]:
        probs_i, idxs_i = torch.topk(p, k=2, dim=-1) # [n, 2], [n, 2]: top 2 biggest posterior probabilities 
        probs_i_m = probs_i[:, 0] # [n, 1]: biggest posterior probabilities (p_max)
        probs_i_s = probs_i[:, 1] # [n, 1]: 2nd biggest posterior probabilities
        idxs_i_m = idxs_i[:, 0] # [n, 1]: indices of biggest posterior probabilities (p_max)
        idxs_i_s = idxs_i[:, 1] # [n, 1]: indices of 2nd biggest posterior probabilities
        
        return probs_i_m, probs_i_s, idxs_i_m, idxs_i_s
    
    def _hard_negatives(self, x: TensorType["n", "d"], x_positive: TensorType["n", "d"]) -> TensorType["n", "d"]: 
        probs = self._posterior(x, self.weights, self.component_distribution) # [n, k]: k posterior probabilites
        probs_i_m, probs_i_s, _, idxs_i_s = self._get_m_s(probs)
         
        x_negative = self.means[idxs_i_s, :] 
        
        cols_idxs = torch.argmax(probs, dim=1)[:, None] # [batch, 1] dimensions of the largest posterior probabilites       
        rows_idxs = torch.arange(0, probs.shape[0], device=x.device)[:, None] # [b, 1]
        mask = torch.ones_like(probs, dtype=torch.bool, device=x.device) # [b, n, k]
        mask[rows_idxs, cols_idxs] = False
        probs_j = probs[mask] # set every biggest row value to zero
            
        conf_n = torch.sigmoid(probs_i_m - probs_i_s) # [b, n]
        conf_d = torch.var(probs_j, dim=-1) # [b, n]
        conf = conf_n / conf_d # [b, n]
        weight = probs_i_s / conf # [b, n]
        weight /= torch.sum(weight) # [b, n]
         
        d = torch.exp(self.sim(x, x_positive) / self.tau) + weight * torch.exp(self.sim(x, x_negative)) # [n]
        n = torch.exp(self.sim(x, x_positive) / self.tau) # [b, n]
        
        loss_cl = - torch.log(n / torch.sum(d)) # [b, 1] 
        
        return loss_cl 
    
    def _false_negatives(self, x: TensorType["n", "d"], x_positive: TensorType["n", "d"]) -> TensorType["n", "d"]:
        n = x.shape[0]
        
        probs = self._posterior(x, self.weights, self.component_distribution) # [b, n, k]: k posterior probabilites
        probs_i_s, probs_i_m, _, _ = self._get_m_s(probs)
        probs_i = torch.topk(probs, k=2, dim=-1).values # [b, n, 2]: top 2 biggest posterior probabilities 
        probs_i_m = probs_i[:, 0] # [b, n, 1]: biggest posterior probabilities (p_max)
        probs_i_s = probs_i[:, 1] # [b, n, 1]: 2nd biggest posterior probabilities
        
        idxs = torch.argmax(probs, dim=-1) # [b, n]: dimension of biggest posterior probabilities
        idxs = idxs.unsqueeze(1) == idxs.unsqueeze(0) # [b, n, n] 
        mask_idxs = ~torch.eye(n, dtype=torch.bool, device=x.device) # [b, n, n]
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
        
        loss_bml = weight / weight_avg * (self.relu(delta + self.bml_alpha) + (self.relu(-delta - self.bml_beta)))
        loss_bml = torch.scatter_add(output_tensor, 0, inverse_indices, loss_bml)
        
        if unique_elements.shape != n: 
            out = torch.zeros(size=(n, ), device=x.device)
            out[unique_elements] = loss_bml
            loss_bml = out 

        return loss_bml
    
    def _entropy_indices(
        self, 
        x: TensorType["n", "d"], 
        p: TensorType["n", "k"], 
        r_idxs: TensorType["n"], 
        c_idxs: TensorType["n"]
        ) -> Tuple[TensorType["1"], ...]:
        
        p = p[r_idxs, c_idxs] # [n]: biggest component of each sample       
        p *= torch.log(p) # [n]: entropy for each sample 
         
        unique_elements, inverse_indices = torch.unique(c_idxs, return_inverse=True) # [n'], [n]
        out = torch.zeros(size=(unique_elements.shape[0], ), device=x.device) # [n']
        
        E_k = torch.zeros(size=(self.k, ), device=x.device) # [k]
        E_k[unique_elements] = -torch.scatter_add(out, 0, inverse_indices, p) # [k]
        
        E_min = torch.min(E_k) # []: distribution with the smallest entropy
        E_max = torch.max(E_k) # []: distribution with the largest entropy
        G_min = torch.argmin(E_k) # []: distribution with the smallest entropy
        G_max = torch.argmax(E_k) # []: distribution with the largest entropy
        
        return E_min, E_max, G_min, G_max
    
    def _zeroth_posterior_update(
        self, 
        x: TensorType["n", "d"]
        ) -> TensorType["n", "k"]:
        
        return self._posterior(x, self.weights, self.component_distribution) # [n, k]
        
    def _first_posterior_update(
        self, 
        x: TensorType["n", "d"], 
        p_zero: TensorType["n", "k"], 
        x_min: TensorType["count_G_min", "d"],
        x_max: TensorType["count_G_max", "d"]
        ) -> TensorType["n", "k"]: 
        
        # Mixture weight of the merged distribution 
        weight_new = (self.count_G_min + self.count_G_max) / x.shape[0]
        weight_buf = 0.0
        
        # Mean vector of the merged distribution 
        mean_new = (torch.sum(x_min, dim=0) + torch.sum(x_max, dim=0)) / (self.count_G_min + self.count_G_max) # [d]
        
        # Exchange posterior probabilities
        p_one = p_zero.clone()
        tmp_min = p_one[self.idxs_min, self.G_min].clone()
        tmp_max = p_one[self.idxs_max, self.G_max].clone()

        p_one[self.idxs_min, self.G_min] = tmp_min
        p_one[self.idxs_max, self.G_max] = tmp_max
        
        # Covariance matrix of the merged distribution 
        num_cov = p_one[:, self.G_max][:, None, None] * ((x - mean_new)[:, :, None] * (x - mean_new)[:, None, :]) # [n, 1, 1] * ([n, d, 1] * .[n, 1, d]) -> [n, d, d]
        num_cov = torch.sum(num_cov, dim=0) # [d, d]
        cov_m = num_cov / (self.count_G_min + self.count_G_max) # [d, d]: new covariance matrix of the Gaussian with the biggest entropy 
        
        # Update weights
        self.weights.data[[self.G_min, self.G_max]] = torch.tensor((weight_buf, weight_new), dtype=self.dtype, device=self.device) # [2]
        
        # Update means
        self.means.data[self.G_max] = mean_new # [d]
        
        # Update covariance matrices
        self.covs.data[self.G_max] = cov_m # [d, d]
        if not self._is_pos_def(self.covs.data[self.G_max]): 
            self.covs.data[self.G_max] = self._to_pos_def(self.covs.data[self.G_max])    

        component_distribution = self._get_mixture_model(means=self.means.data.clone(), covs=self.covs.data.clone())
        
        # Recalculate posterior probabilities of the samples belonging to the updated posterior probability
        p_one[self.idxs_min] = self._posterior(x=x_min, weights=self.weights.data, distribution=component_distribution) # [n, k]
        p_one[self.idxs_max] = self._posterior(x=x_max, weights=self.weights.data, distribution=component_distribution) # [n, k]
        
        return p_one

    def _second_posterior_update(
        self, 
        x: TensorType["n", "d"], 
        p_one: TensorType["n", "k"]
        ) -> Tuple[TensorType["*"], ...]: 
                 
        mean_one = self.means[self.G_max].data.clone()    
        cov_one = self.covs[self.G_max].data.clone()
        
        dim_max_var = torch.argmax(torch.diagonal(cov_one)).item() # []: dimension with the biggest variance

        idxs = torch.hstack((self.idxs_min, self.idxs_max)) # [count_G_min+count_G_max, d]: all data points belonging to the merged distribution 
        idxs = idxs.flatten()
        
        swap_mask = x[idxs, dim_max_var] <= mean_one[dim_max_var] 
        stay_mask = ~swap_mask
        swap_idxs = torch.sort(idxs[swap_mask]).values # [count_G_min_two]: indices of all samples belonging to the new smallest distribution 
        stay_idxs = torch.sort(idxs[stay_mask]).values # [count_G_max_two]: indices of all samples belonging to the new biggest distribution 
         
        count_new_min = swap_idxs.numel()
        count_new_max = stay_idxs.numel() 
        
        wandb.log({"diff_min": count_new_min - self.count_G_min})
        wandb.log({"diff_max": count_new_max - self.count_G_max})
        
        if count_new_min <= 0: 
            raise ValueError(f"There are no samples to swap!! {count_new_min}")       
        if count_new_max <= 0: 
            raise ValueError(f"There are no samples to swap!! {count_new_max}")
        
        # Exchange posterior probabilities
        p_two = p_one.clone()
        tmp_min = p_two[stay_idxs, self.G_min].clone()
        tmp_max = p_two[stay_idxs, self.G_max].clone()

        p_two[stay_idxs, self.G_min] = tmp_max
        p_two[stay_idxs, self.G_max] = tmp_min

        x_min_two = x[swap_idxs]
        x_max_two = x[stay_idxs]

        # Update the mixture weights of the two new Gaussian distributions
        weight_min_two = torch.tensor(count_new_min / x.shape[0]) # []: new weight of the Gaussian with the past smallest entropy 
        weight_max_two = torch.tensor(count_new_max / x.shape[0]) # []: new weight of the Gaussian with the past biggest entropy 
        self.weights.data[[self.G_min, self.G_max]] = torch.hstack((weight_min_two, weight_max_two)).to(self.dtype).to(self.device) # [2]
        
        # update the mean vectorss of the two new Gaussian distributions
        mean_min_two = torch.sum(x_min_two, dim=0) / count_new_min # [d]: new mean vector of the Gaussian with the biggest entropy                
        mean_max_two = torch.sum(x_max_two, dim=0) / count_new_max # [d]: new mean vector of the Gaussian with the past biggest entropy 
        self.means.data[[self.G_min, self.G_max]] = torch.vstack((mean_min_two, mean_one)) # [2, d]
        
        cov_min_two = p_two[:, self.G_min][:, None, None] * ((x - mean_min_two).unsqueeze(-1) * (x - mean_min_two).unsqueeze(-2)) # [n, d, 1] * [n, 1, d] -> [n, d, d]
        cov_min_two = torch.sum(cov_min_two, dim=0) / count_new_min # [d, d]
        cov_max_two = p_two[:, self.G_max][:, None, None] * ((x - mean_max_two).unsqueeze(-1) * (x - mean_max_two).unsqueeze(-2)) # [n, d, 1] * [n, 1, d] -> [n, d, d]
        cov_max_two = torch.sum(cov_max_two, dim=0) / count_new_max # [d, d]
        self.covs.data[[self.G_min, self.G_max]] = torch.vstack((cov_min_two[None, ...], cov_max_two[None, ...]))
        
        if not self._is_pos_def(self.covs.data[[self.G_min, self.G_max]]): 
            self.covs.data[[self.G_min, self.G_max]] = self._to_pos_def(self.covs.data[[self.G_min, self.G_max]]) 
            
        component_distribution = self._get_mixture_model(means=self.means.data, covs=self.covs.data)
        
        p_min_two = self._posterior(x=x_min_two, weights=self.weights.data, distribution=component_distribution) # [count_G_min, k]
        p_max_two = self._posterior(x=x_max_two, weights=self.weights.data, distribution=component_distribution) # [count_G_max, k]

        p_one[swap_idxs] = p_min_two
        p_one[stay_idxs] = p_max_two
        
        return p_one, x_min_two, x_max_two
    
    def _mahalanobis_distance(self, x, mean, cov): 
        distances = (x - mean)[:, None, :] @ torch.inverse(cov)[None, :, :] @ (x - mean)[:, :, None] #  [n, 1, d] * [1, d, d] * [n, d, 1] -> [n, 1, 1]
        distances = torch.sqrt(distances) # [n, 1, 1]
        return distances.view(-1) # [n]
    
    def _is_convergence(self, x_min_zero, x_max_zero, x_min_two, x_max_two, means_zero, covs_zero) -> torch.Tensor.bool:  
        one_left = torch.sum(torch.exp(-self._mahalanobis_distance(x_min_two, self.means.data[self.G_min], self.covs.data[self.G_min])))
        one_right = torch.sum(torch.exp(-self._mahalanobis_distance(x_min_zero, means_zero[self.G_min], covs_zero[self.G_min])))
        one = one_left >= one_right # (Equation 21a)
        # print(f"one:{one}")
        
        two_left = torch.sum(-torch.exp(-self._mahalanobis_distance(x_max_two, self.means.data[self.G_max], self.covs.data[self.G_max])))
        two_right = torch.sum(-torch.exp(-self._mahalanobis_distance(x_max_zero, means_zero[self.G_max], covs_zero[self.G_max])))
        two = two_left >= two_right # (Equation 21b)
        # print(f"two:{two}")

        three = torch.det(covs_zero[self.G_min]) >= torch.det(self.covs.data[self.G_min]) # (Equation 21c)
        # print(f"three:{three}")

        four = torch.det(covs_zero[self.G_max]) >= torch.det(self.covs.data[self.G_max]) # (Equation 21cd)
        # print(f"four:{four}")

        five_left = torch.sum(torch.exp(-self._mahalanobis_distance(x_min_two, self.means.data[self.G_min], self.covs.data[self.G_min])))
        five_left /= torch.sqrt(torch.det(self.covs.data[self.G_min]))
        five_right = torch.sum(torch.exp(-self._mahalanobis_distance(x_max_two, self.means.data[self.G_max], self.covs.data[self.G_max])))
        five_right /= torch.sqrt(torch.det(self.covs.data[self.G_max]))
        five = five_left >= five_right # (Equation 23)
        # print(f"five:{five}")

        _is_convergence = torch.stack([one, two, three, four, five])
        return torch.all(_is_convergence).item()
    
    def _log_likelihood(self, x, p, means, covs):
        labels = torch.argmax(p, dim=-1) # [n]
        log_2_pi = torch.log(torch.tensor(2 * np.pi, dtype=self.dtype, device=self.device))
        constant_term = -0.5 * self.d * log_2_pi
        
        log_likelihood = torch.tensor(0.0, dtype=self.dtype, device=self.device)
        
        for k in range(self.k): 
            idxs = torch.nonzero(labels == k).view(-1)
            if idxs.numel() == 0: 
                continue 
            
            x_k = x[idxs]          
            L = torch.linalg.cholesky(covs[k])
            log_det_cov_k = 2.0 * torch.sum(torch.log(torch.diagonal(L)))            
            
            d_k = self._mahalanobis_distance(x_k, means[k], covs[k])
            
            cluster_ll = torch.sum(-0.5 * d_k - 0.5 * log_det_cov_k + constant_term)
            log_likelihood += cluster_ll
        
        return log_likelihood / x.shape[0]

    def _update_gmm(self, x: TensorType["n", "d"])  -> None: 
        columns = [f"dim_{i}" for i in range(self.d)] + ["label"]
        
        # Initial posterior update: hard assignments -> soft assignments 
        p_zero = self._zeroth_posterior_update(x) # [n, k]: posterior probability of each sample belonging to Gaussian component k 
        
        # General computations         
        r_idxs = torch.arange(x.shape[0], device=x.device) # [n]
        c_idxs = torch.argmax(p_zero, dim=-1) # [n]: Gaussian component k each sample most likely belongs to 
        
        # Initial mixture weights update
        counts = torch.zeros(size=(self.k, ), dtype=self.dtype, device=self.device)
        counts.scatter_add_(0, c_idxs, torch.ones_like(c_idxs, dtype=self.dtype))
        self.weights.data = counts / x.shape[0] # [k]
        
        # Posterior update loop: 
        i = 0
        while(True):  
            # Entropy-related calculation
            E_min, E_max, G_min, G_max = self._entropy_indices(x, p_zero, r_idxs, c_idxs) # [], [], [], []: smallest/ biggest entropy, indices of the Gaussians with the smallest/ biggest entropies
            self.G_min = G_min.item() 
            self.G_max = G_max.item() 
            
            self.idxs_min = torch.nonzero(self.G_min == c_idxs, as_tuple=False).view(-1) # [count_G_min]: indices of all samples belonging to the Gaussian with the smallest entropy
            self.idxs_max = torch.nonzero(self.G_max == c_idxs, as_tuple=False).view(-1) # [count_G_max]: indices of all samples belonging to the Gaussian with the biggest entropy

            self.count_G_min = self.idxs_min.numel() # []: number of samples in the Gaussian with the smallest entropy 
            self.count_G_max = self.idxs_max.numel() # []: number of samples in the Gaussian with the biggest entropy 
            
            x_min_zero = x[self.idxs_min] # [count_G_min, d]: samples belonging to the Gaussian with the smallest entropy
            x_max_zero = x[self.idxs_max] # [count_G_max, d]: samples belonging to the Gaussian with the biggest entropy
            
            means_zero = self.means.data.clone()
            covs_zero = self.covs.data.clone()
        
            if E_min <= E_max: 
                # 1st posterior update
                p_one = self._first_posterior_update(x, p_zero, x_min_zero, x_max_zero)
            else: 
                print("Failure")
                break
            
            # 2nd posterior update
            p_two, x_min_two, x_max_two = self._second_posterior_update(x, p_one)
            
            if i == 100: 
                labels = torch.argmax(p_two, dim=-1).view(-1).tolist()
                data = [list(vec)+[lbl] for vec, lbl in zip(x, labels)]
                
                table = wandb.Table(columns=columns, data=data)
                wandb.log({"high_dim_emb": table})

                    
            # Check for convergence
            is_convergence = self._is_convergence(x_min_zero, x_max_zero, x_min_two, x_max_two, means_zero, covs_zero)
            log_likelihood = self._log_likelihood(x, p_two, self.means.data, self.covs.data)

            wandb.log({"log_likelihood": log_likelihood})

            if is_convergence: 
                return True 
            elif i >= self.max_iter: 
                return False
            else: 
                p_zero = p_two 
                i += 1
    
    def _update_encoder(self, x, x_plus):
        loss_cl = self._hard_negatives(x, x_plus)
        loss_bml = self._false_negatives(x, x_plus)
        return loss_cl, loss_bml
        
    def forward(self, x: TensorType["b", "n", "d"], x_plus: Optional[TensorType["b", "n", "d"]]=None) -> TensorType["*"]:
        assert x.numel() != 0, "Inputs cannot be empty!"
        assert x_plus.numel() != 0, "Positive pairs cannot be empty!"

        self.dtype = x.dtype
        self.device = x.device
        self.component_distribution = self._get_mixture_model(means=self.means, covs=self.covs)
        
        _gmm_update = self.update_gmm(x)
        
        if not _gmm_update: 
            raise ValueError 
        
        loss_cl, loss_bml = self._update_encoder(x, x_plus) 
        return loss_cl, loss_bml