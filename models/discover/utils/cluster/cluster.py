import numpy as np 
from tqdm import tqdm 
from termcolor import colored
from omegaconf import DictConfig
from typing import Dict, Optional, Tuple 

import torch
import torch.nn as nn 
import torch.nn.functional as F
import torch.distributions as D
from torchtyping import TensorType
from torch.linalg import cholesky, cholesky_ex, eigvalsh, solve_triangular

import wandb

class KMeans(nn.Module):
    def __init__(self, 
        k: int, 
        d: int, 
        max_iter: int,
        rtol: float
        ) -> None:
        super().__init__()
        
        if k <= 0: raise ValueError(colored(f"The number of clusters has to be positive, got {k}!", "red"))
        if d <= 0: raise ValueError(colored(f"The dimension of data points has to be positive, got {d}!", "red"))
        if max_iter <= 0: raise ValueError(colored(f"The number of fit iterations has to be positive, got {max_iter}!", "red"))
        if rtol <= 0: raise ValueError(colored(f"The convergence tolerance has to be positive, got {rtol}!", "red"))

        self.k = k
        self.d = d 
        self.max_iter = max_iter
        self.rtol = rtol
        
        self.register_buffer("weights", torch.empty(size=(self.k, ), dtype=torch.float32)) # [k, 1]
        self.register_buffer("means", torch.empty(size=(self.k, self.d), dtype=torch.float32)) # [k, d]
        self.register_buffer("covs", torch.empty(size=(self.k, self.d, self.d), dtype=torch.float32)) # [k, d, d]
        
        self.is_fitted = False
    
    @torch.no_grad() 
    def _compute_distances(self, x: TensorType["n", "d"]) -> TensorType["n", "k"]:
          # ||X - C||^2 = ||X||^2 + ||C||^2 - 2*X*C.T  
        x_norm = (x ** 2).sum(dim=1, keepdim=True) # [n, 1]
        c_norm = (self.means ** 2).sum(dim=1, keepdim=True).T # [1, k]
        distances = x_norm + c_norm - 2 * x @ self.means.T # [n, k]
        return distances
    
    @torch.no_grad() 
    def _fit(self, x: TensorType["n", "d"]) -> "KMeans":
        n, _ = x.shape
        assert n >= self.k, f"Need at least {self.k} points, got {n}"
        
        # Random init from data 
        rand_idxs = torch.randperm(n, device=x.device)[:self.k]
        self.means = x.clone()[rand_idxs] # [k, d]
                
        for _ in range(self.max_iter):
            old_means = self.means.clone() 
            distances = self._compute_distances(x) # [n, k] 
            labels = torch.argmin(distances, dim=1) # [n]: hard cluster assignments
            one_hot = F.one_hot(labels, num_classes=self.k).to(x.dtype) # [n, k]: one-hot centroid mapping matrix (one in column of assigned cluster; zero elsewhere)      
            counts = torch.clamp(torch.sum(one_hot, dim=0, keepdim=True).T, min=1e-8) # [k, 1] # count points in each cluster (prevent division by zero with small epsilon)
            self.means.copy_((one_hot.T @ x) / counts) # [k, n] @ [n, d] / [k, 1]
            
            # Check for convergence by tracking the shift of the centroids
            if torch.allclose(self.means, old_means, rtol=self.rtol): 
                break

        self.is_fitted = True
        return self
    
    @torch.no_grad()
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
    
    @torch.no_grad()
    def forward(self, x: TensorType["n", "d"]) -> Dict[str,TensorType["*"]]:
        if not self.is_fitted:
            self._fit(x)
            
        distances = self._compute_distances(x) # [n, k]
        labels = torch.argmin(distances, dim=1) # [n]
        counts = torch.zeros(self.k, dtype=torch.long, device=x.device) # [k]
        counts.scatter_add_(0, labels, torch.ones_like(labels)) #[k]

        self.weights = counts.float() / x.shape[0] # [k]: sum to 1 
        self.covs = self._get_covariance(x, labels) # [k, d, d]
        
        return {
            "weights": self.weights, # [k]
            "means": self.means, # [k, d]
            "covs": self.covs, # [k, d, d]
            "labels": labels # [n]
        } 


class RGMM(nn.Module): 
    def __init__(
        self, 
        k: int, # number of Gaussian distributions  
        d: int, # dimension of data and parameters of Gaussian distributions 
        jitter_noise: float, 
        shift_scale: float, 
        shift_bias: float, 
        max_iter: int,
        bml_weight: float, 
        bml_alpha: float, 
        bml_beta: float, 
        sim: DictConfig # parameters of cosine similarity
        ) -> None: 
        super().__init__()
        
        self.k = k 
        self.d = d
        self.jitter_noise = jitter_noise
        self.shift_scale = shift_scale 
        self.shift_bias = shift_bias 
        self.max_iter = max_iter
        self.bml_weight = bml_weight
        self.bml_alpha = bml_alpha 
        self.bml_beta = bml_beta 
        
        self.register_buffer("weights", torch.ones(self.k, dtype=torch.float32) / self.k) # [k]
        self.register_buffer("means", torch.randn(size=(self.k, self.d), dtype=torch.float32)) # [k, d]
        self.register_buffer("covs",  torch.eye(n=self.d).unsqueeze(0).repeat(self.k, 1, 1)) # [k, d, d]
        
        self.tau = nn.Parameter(data=torch.randn((1, ), dtype=torch.float32)) # [1]
        self.lam = nn.Parameter(data=torch.randn(size=(1, ), dtype=torch.float32)) # [1]
        
        self.sim = nn.CosineSimilarity(**sim)
        
        self.component_distribution, self.dtype, self.device, self.logger = [None]*4
        self.G_min, self.G_max, self.count_G_min, self.count_G_max, self.idxs_min, self.idxs_max, self.n = [0]*7 
        
        self.relu = nn.ReLU()
    
    def _get_mixture_model(self, means: TensorType["k", "d"], covs: TensorType["k", "d", "d"]) -> D.MultivariateNormal:
        """Create a mixture model of multivariate Gaussian distributions.
        
        Args:
            means: Mean vectors of shape [k, d] 
            covs: Covariance matrices of shape [k, d, d]
        
        Returns:
            D.MultivariateNormal: Multivariate normal distribution.
        """
        if not self._is_pos_def(covs):
            covs = self._to_pos_def(covs)
        
        component_distribution = D.MultivariateNormal(loc=means, covariance_matrix=covs)
        
        return component_distribution
    
    def _is_pos_def(self, A: TensorType["k", "d", "d"]) -> bool:
        """Check if covariance matrices are positive definite.
        
        Args:
            A: Covariance matrices of shape [k, d, d].
        
        Returns:
            bool: True if all matrices are positive definite, False otherwise.
        """
        if not torch.allclose(A, A.transpose(-1, -2)): # 1st: ensure that the matrix is symmetrix
            return False 
        
        _, info = cholesky_ex(A) # 2nd: Cholesky decomposition

        return bool(torch.all(info == 0))

    def _to_pos_def(self, A: TensorType["k", "d", "d"]) -> TensorType["k", "d", "d"]:
        """Convert covariance matrices to positive definite form.
        
        Args:
            A: Covariance matrices of shape [k, d, d].
        
        Returns:
            TensorType: Positive definite covariance matrices of shape [k, d, d].
        """
        eye = torch.eye(self.d, dtype=self.dtype, device=self.device)
        
        # -> Symmetric covariance matrices 
        A = (A + A.transpose(-1, -2)) / 2 
        
        # -> Positive definite covariance matrices 
        A = A.to(torch.float64)
        A = A + self.jitter_noise * eye
        
        eigenvals = eigvalsh(A) # eigendecomposition
        min_eig = eigenvals.amin(dim=-1)  
        shift = torch.clamp(-min_eig, min=0) * self.shift_scale + self.shift_bias # eigenvalue clipping
        
        # -> Reconstruction
        A = A + shift[..., None, None] * eye
        A = A.to(self.dtype)

        return A

    def _posterior(
        self,
        x: TensorType["n", "d"],
        weights: TensorType["k"], 
        distribution: D.MultivariateNormal
    ) -> TensorType["n", "k"]:
        """Compute posterior probabilities using Bayes' rule
        
        Args:
            x: Input data
            weights: Mixture weights 
            distribution: Multivariate normal distribution.
        
        Returns:
            Posterior probabilities
        """
        log_probs = distribution.log_prob(x.unsqueeze(1)) # [n, k]: unweighted log-density
        log_weights = torch.log(weights) # [k]
        log_joint = log_probs + log_weights # [n, k]: log(w_k * log(N(x | mu_k, Sigma_k)) = log(w_k) + log(N(x | mu_k, Sigma_k))
        log_posterior = log_joint - torch.logsumexp(log_joint, dim=-1, keepdim=True) # [n, k]: normalize over k 
        log_posterior = log_posterior.exp()
        
        return log_posterior # [n, k]: posterior probabilities p_{i,k} 
    
    def _confidence(self, x: TensorType["n", "d"]) -> TensorType["n"]:
        """Compute confidence scores for samples.
        
        Args:
            x: Input data of shape [n, d].
        
        Returns:
            TensorType: Confidence scores of shape [n].
        """
        n, _ = x.shape
        probs = self._posterior(x, self.weights, self.component_distribution) # [n, k]: k posterior probabilites
        vals = torch.topk(probs, k=2, dim=-1).values # [n, 2]
        
        rows = torch.arange(n, dtype=torch.long, device=self.device) # [n]
        cols = torch.argmax(probs, dim=1) #  [n]
        
        mask = torch.ones_like(probs, dtype=torch.bool, device=self.device) # [n, k] 
        mask[rows, cols] = False # [n, k] 
        probs = probs[mask].view(n, self.k-1) # [b, n, k-1]
        
        confidence = torch.sigmoid(vals[:, 0]-vals[:, 1]) / torch.sigmoid(torch.var(probs, dim=-1)) # [n]
        
        return confidence
    
    def _get_m_s(self, p: TensorType["n", "k"]) -> TensorType["n", "2"]:
        """Extract top 2 posterior probabilities and their indices.
        
        Args:
            p: Posterior probabilities of shape [n, k].
        
        Returns:
            Tuple of (max_probs [n], second_max_probs [n], max_indices [n], second_max_indices [n]).
        """
        probs_i, idxs_i = torch.topk(p, k=2, dim=-1) # [n, 2], [n, 2]: top 2 biggest posterior probabilities 
        probs_i_m = probs_i[:, 0] # [n, 1]: biggest posterior probabilities (p_max)
        probs_i_s = probs_i[:, 1] # [n, 1]: 2nd biggest posterior probabilities
        idxs_i_m = idxs_i[:, 0] # [n, 1]: indices of biggest posterior probabilities (p_max)
        idxs_i_s = idxs_i[:, 1] # [n, 1]: indices of 2nd biggest posterior probabilities
        
        return probs_i_m, probs_i_s, idxs_i_m, idxs_i_s
    
    def _hard_negatives(self, x: TensorType["n", "d"], x_plus: TensorType["n", "d"]) -> TensorType[""]:
        """Find hard negatives, and compute contrastive loss.
        
        Args:
            x: Input data of shape [n, d].
            x_positive: Positive pairs of shape [n, d].
        
        Returns:
            torch.tensor: Contrastive loss.
        """
        probs = self._posterior(x, self.weights, self.component_distribution) # [n, k]: k posterior probabilites
        probs_i_m, probs_i_s, _, idxs_i_s = self._get_m_s(probs)
         
        x_negative = self.means.clone()[idxs_i_s, :] 
        
        cols_idxs = torch.argmax(probs, dim=1).unsqueeze(-1) # [batch, 1] dimensions of the largest posterior probabilites       
        rows_idxs = torch.arange(0, probs.shape[0], device=x.device).unsqueeze(-1) # [b, 1]
        mask = torch.ones_like(probs, dtype=torch.bool, device=x.device) # [b, n, k]
        mask[rows_idxs, cols_idxs] = False
        probs_j = probs[mask] # set every biggest row value to zero
            
        conf_n = torch.sigmoid(probs_i_m - probs_i_s) # [b, n]
        conf_d = torch.var(probs_j, dim=-1) # [b, n]
        conf = conf_n / conf_d # [b, n]
        weight = probs_i_s / conf # [b, n]
        weight = weight / torch.sum(weight) # [b, n]
         
        nom = torch.exp(self.sim(x, x_plus) / self.tau) # [b, n]
        denom = torch.exp(self.sim(x, x_plus) / self.tau) + weight * torch.exp(self.sim(x, x_negative) / self.tau) # [n]

        loss_cl = -torch.log(nom / torch.sum(denom)) # [b, 1] 
        loss_cl = torch.mean(loss_cl) 
         
        return loss_cl 
    
    def _false_negatives(self, x: TensorType["n", "d"], x_positive: TensorType["n", "d"]) -> TensorType[""]:
        """Compute loss accounting for false negatives in clustering.
        
        Args:
            x: Input data of shape [n, d].
            x_positive: Positive pairs of shape [n, d].
        
        Returns:
            torch.tensor: Loss for false negatives.
        """
        n, _ = x.shape
        
        probs = self._posterior(x, self.weights, self.component_distribution) # [n, k]: k posterior probabilites
        probs_i_m, probs_i_s, _, _ = self._get_m_s(probs)
        
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
        counts = torch.zeros(size=(torch.unique(rows_idxs).numel(), ), device=x.device)
        counts = torch.scatter_add(counts, 0, inverse_indices, torch.ones_like(weight, device=x.device))
        output_tensor = torch.zeros(size=(torch.unique(rows_idxs).numel(), ), device=x.device)
        weight_avg = torch.scatter_add(output_tensor, 0, inverse_indices, weight) 
        weight_avg = weight_avg / counts
        weight_avg = weight_avg[inverse_indices, ...]
        
        x_j = x.clone()[cols_idxs, ...]
        x = x[rows_idxs, ...]
        x_positive = x_positive[rows_idxs, ...]
        delta = self.sim(x, x_j) - self.sim(x, x_positive) 
        
        loss_bml = weight / weight_avg * (self.relu(delta + self.bml_alpha) + (self.relu(-delta - self.bml_beta)))
        loss_bml = torch.scatter_add(output_tensor, 0, inverse_indices, loss_bml)
        
        if unique_elements.numel() != n: 
            out = torch.zeros(size=(n, ), device=x.device)
            out[unique_elements] = loss_bml
            loss_bml = out 
        
        loss_bml = torch.mean(loss_bml)
        return loss_bml
    
    @torch.no_grad()
    def _entropy_indices(
        self, 
        x: TensorType["n", "d"], 
        p: TensorType["n", "k"], 
        r_idxs: TensorType["n"], 
        c_idxs: TensorType["n"]
        ) -> Tuple[TensorType[""], TensorType[""]]:
        """Compute entropy and identify components with min/max entropy.
        
        Args:
            x: Input data of shape [n, d].
            p: Posterior probabilities of shape [n, k].
            r_idxs: Row indices of shape [n].
            c_idxs: Column indices (component assignment) of shape [n].
        
        Returns:
            Tuple of (G_min, G_max).
        """
        p_max = p.clone()[r_idxs, c_idxs] # [n]: biggest likelihood of each sample      
        p_max = p_max.clamp_min(min=1e-12)
        p_max = p_max * torch.log(p_max) # [n]: entropy for each sample 
         
        unique_elements, inverse_indices = torch.unique(c_idxs, return_inverse=True) # [n'], [n]
        out = torch.zeros(size=(unique_elements.shape[0], ), device=x.device) # [n']
        
        E_k = torch.full(size=(self.k, ), fill_value=torch.tensor(float("nan")), device=x.device) 
        E_k[unique_elements] = -torch.scatter_add(out, 0, inverse_indices, p_max) # [k]
        
        G_min = torch.argmin(torch.nan_to_num(E_k, nan=float("inf"))) # []: distribution with the smallest entropy
        G_max = torch.argmax(torch.nan_to_num(E_k, nan=float("-inf"))) # []: distribution with the largest entropy
    
        return G_min, G_max
    
    @torch.no_grad()
    def _zeroth_posterior_update(self, x: TensorType["n", "d"]) -> TensorType["n", "k"]:
        """Compute initial posterior probabilities
        
        Args:
            x: Input data
        
        Returns:
            Posterior probabilities
        """
        return self._posterior(x, self.weights, self.component_distribution) # [n, k]
    
    @torch.no_grad()   
    def _first_posterior_update(
        self, 
        x: TensorType["n", "d"], 
        p_zero: TensorType["n", "k"], 
        x_min: TensorType["count_G_min", "d"],
        x_max: TensorType["count_G_max", "d"]
        ) -> TensorType["n", "k"]:
        """First posterior update: merge components with min and max entropy.
        
        Args:
            x: Input data of shape [n, d].
            p_zero: Initial posterior probabilities of shape [n, k].
            x_min: Data belonging to min entropy component.
            x_max: Data belonging to max entropy component.
        
        Returns:
            TensorType: Updated posterior probabilities of shape [n, k].
        """
        n, _ = x.shape
        
        # Mixture weight of the merged distribution 
        weight_new = (self.count_G_min + self.count_G_max) / n
        weight_buf = 0.0
        
        # Mean vector of the merged distribution
        x_new = torch.vstack(tensors=(x_min, x_max))
        mean_one = torch.sum(x_new, dim=0) / (self.count_G_min + self.count_G_max) # [d]
        
        # Exchange posterior probabilities
        p_one = p_zero.clone()
        tmp_min = p_one.clone()[self.idxs_min, self.G_min]
        tmp_max = p_one.clone()[self.idxs_min, self.G_max]
        p_one[self.idxs_min, self.G_min] = tmp_max
        p_one[self.idxs_min, self.G_max] = tmp_min
        
        # Covariance matrix of the merged distribution 
        diff = x - mean_one
        num_cov = p_one[:, self.G_max][:, None, None] * (diff.unsqueeze(-1) * diff.unsqueeze(-2)) # [n, 1, 1] * ([n, d, 1] * [n, 1, d]) -> [n, d, d]
        num_cov = torch.sum(num_cov, dim=0) # [d, d]
        cov_m = num_cov / (self.count_G_min + self.count_G_max) # [d, d]: new covariance matrix of the Gaussian with the biggest entropy 
        
        # Update weights
        self.weights[[self.G_min, self.G_max]] = torch.tensor((weight_buf, weight_new), dtype=self.dtype, device=self.device) # [2]
        # Update means
        self.means[self.G_max] = mean_one # [d]
        # Update covariance matrices
        self.covs[self.G_max] = cov_m # [d, d]
        if not self._is_pos_def(self.covs[self.G_max]): 
            self.covs = self._to_pos_def(self.covs)

        component_distribution = self._get_mixture_model(means=self.means, covs=self.covs)
        
        # Recalculate posterior probabilities of the samples belonging to the updated posterior probability
        p_one[self.idxs_min] = self._posterior(x=x_min, weights=self.weights, distribution=component_distribution) # [n, k]
        p_one[self.idxs_max] = self._posterior(x=x_max, weights=self.weights, distribution=component_distribution) # [n, k]
        
        return p_one
    
    @torch.no_grad()
    def _second_posterior_update(
        self, 
        x: TensorType["n", "d"], 
        p_one: TensorType["n", "k"]
        ) -> Tuple[TensorType["*"], ...]:
        """Second posterior update: Split merged component along max variance dimension.
        Args:
            x: Input data of shape [n, d].
            p_one: Posterior probabilities after first update of shape [n, k].
        Returns:
            Tuple of (updated_posteriors, x_min, x_max).
        """
        n, _ = x.shape
                 
        mean_one = self.means.clone()[self.G_max] # mean of merged distribution   
        cov_one = self.covs.clone()[self.G_max] # covariance of merged distribution
        
        dim_max_var = torch.argmax(torch.diagonal(cov_one)).item() # []: dimension with the biggest variance

        idxs = torch.hstack((self.idxs_min, self.idxs_max)).flatten() # [count_G_min+count_G_max]: all data points belonging to the merged distribution 
        
        stay_mask = x[idxs, dim_max_var] <= mean_one[dim_max_var] 
        swap_mask = ~stay_mask
        stay_idxs = torch.sort(idxs[stay_mask]).values # [count_G_max_two]: indices  of all samples belonging to the new distribution with the biggest entropy
        swap_idxs = torch.sort(idxs[swap_mask]).values # [count_G_min_two]: indices of all samples belonging to the new distribution with the smallest entropy 
         
        count_new_min = swap_idxs.numel() # number of all samples belonging to the new distribution with the smallest entropy  
        count_new_max = stay_idxs.numel() # number of all samples belonging to the new distribution with the biggest entropy 

        if count_new_min <= 0: 
            raise ValueError(f"There are no samples to swap!! {count_new_min}") # TODO: Implement fallback    
        if count_new_max <= 0: 
            raise ValueError(f"There are no samples to swap!! {count_new_max}") # TODO: Implement fallback 
        
        # Exchange posterior probabilities
        p_two = p_one.clone()
        tmp_min = p_two.clone()[swap_idxs, self.G_min]
        tmp_max = p_two.clone()[swap_idxs, self.G_max]
        p_two[swap_idxs, self.G_min] = tmp_max 
        p_two[swap_idxs, self.G_max] = tmp_min 

        x_min_two = x.clone()[swap_idxs] # all samples belonging to the new distribution with the smallest entropy 
        x_max_two = x.clone()[stay_idxs] #all samples belonging to the new distribution with the biggest entropy 

        # Update the mixture weights of the two new Gaussian distributions
        weight_min_two = torch.tensor(count_new_min / n) # []: new weight of the Gaussian with the past smallest entropy 
        weight_max_two = torch.tensor(count_new_max / n) # []: new weight of the Gaussian with the past biggest entropy 
        self.weights[[self.G_min, self.G_max]] = torch.hstack((weight_min_two, weight_max_two)).to(self.dtype).to(self.device) # [2]
        
        # update the mean vectorss of the two new Gaussian distributions
        mean_min_two = torch.sum(x_min_two, dim=0) / count_new_min # [d]: new mean vector of the Gaussian with the biggest entropy                
        mean_max_two = torch.sum(x_max_two, dim=0) / count_new_max # [d]: new mean vector of the Gaussian with the past biggest entropy 
        self.means[[self.G_min, self.G_max]] = torch.vstack((mean_min_two, mean_max_two)) # [2, d]
        
        diff = x - mean_min_two
        cov_min_two = p_two[:, self.G_min][:, None, None] * (diff.unsqueeze(-1) * diff.unsqueeze(-2)) # [n, d, 1] * [n, 1, d] -> [n, d, d]
        cov_min_two = torch.sum(cov_min_two, dim=0) / count_new_min # [d, d]
        cov_max_two = p_two[:, self.G_max][:, None, None] * (diff.unsqueeze(-1) * diff.unsqueeze(-2)) # [n, d, 1] * [n, 1, d] -> [n, d, d]
        cov_max_two = torch.sum(cov_max_two, dim=0) / count_new_max # [d, d]
        self.covs[[self.G_min, self.G_max]] = torch.vstack((cov_min_two[None, ...], cov_max_two[None, ...]))
        
        if not self._is_pos_def(self.covs): 
            self.covs = self._to_pos_def(self.covs)
            
        component_distribution = self._get_mixture_model(means=self.means, covs=self.covs)
        
        p_one[swap_idxs] = self._posterior(x=x_min_two, weights=self.weights, distribution=component_distribution) # [count_G_min, k]
        p_one[stay_idxs] = self._posterior(x=x_max_two, weights=self.weights, distribution=component_distribution) # [count_G_max, k]
        
        return p_one, x_min_two, x_max_two
    
    @torch.no_grad()
    def _mahalanobis_distance(
        self, 
        x: TensorType["n", "d"],
        mean: TensorType["d"], 
        cov: TensorType["d", "d"], 
        L: Optional[TensorType["d", "d"]]=None
        ) -> TensorType["n"]:
        """Compute Mahalanobis distance between samples and a Gaussian component.
        
        Args:
            x: Input samples of shape [n, d].
            mean: Mean of the Gaussian component of shape [d].
            cov: Covariance matrix of the Gaussian component: of shape [d, d]
            L: Precomputed Cholesky factor of cov; if None, computed internally from cov
        
        Returns:
            torch.tensor: Mahalanobis distances of shape [n]
        """
        if L is None: 
            L = cholesky(cov) # [d, d]: cov = LL^T 
        
        # diff^T @ (LL^T)^-1 @ diff = diff^T @ L^-1^T @ L^⁻1 @ diff = (L^-1 @ diff)^T @ L^-1 @ diff = z^T @ z = ||z|| ** 2 
        # solve_triangular: A * X = B -> B^-1 * A = X^-1
        
        diff = (x-mean).unsqueeze(-1) # [n, d, 1]: batch of n columns vectors
        z = solve_triangular(L, diff, upper=False) # [n, n] @ [n, d, 1] -> [n, d, 1]: solve z = L^-1 @ diff 
        sq_distance = (z.squeeze(-1) ** 2).sum(-1) # [n]: ||z|| ** 2
        distances = torch.sqrt(torch.clamp_min(sq_distance, min=0.0)) # [n]: ||z||
        
        return distances 
    
    @torch.no_grad()
    def _is_convergence(
        self, 
        x_min_zero: TensorType["n_min_zero", "d"], 
        x_max_zero: TensorType["n_max_zero", "d"], 
        x_min_two: TensorType["n_min_two", "d"], 
        x_max_two: TensorType["n_max_two", "d"], 
        means_zero: TensorType["n", "d"],  
        covs_zero: TensorType["n", "d", "d"]
        ) -> torch.Tensor.bool:
        """Check convergence criteria for the GMM update.
        
        Args:
            x_min_zero: Min entropy data before splits of shape [n_min_zero, d].
            x_max_zero: Max entropy data before splits of shape [n_max_zero, d].
            x_min_two: Min entropy data after splits of shape [n_min_two, d].
            x_max_two: Max entropy data after splits of shape [n_max_two, d].
            means_zero: Original means before updates of shape [n, d, d].
            covs_zero: Original covariances before updates of shape [n, d, d].
        
        Returns:
            torch.tensor.bool: True if convergence criteria are satisfied.
        """
        one_left = torch.sum(torch.exp(-self._mahalanobis_distance(x_min_two, self.means[self.G_min], self.covs[self.G_min])))
        one_right = torch.sum(torch.exp(-self._mahalanobis_distance(x_min_zero, means_zero[self.G_min], covs_zero[self.G_min])))
        one = one_left >= one_right # (Equation 21a)
        print(f"one: {one}")
        
        two_left = torch.sum(-torch.exp(-self._mahalanobis_distance(x_max_two, self.means[self.G_max], self.covs[self.G_max])))
        two_right = torch.sum(-torch.exp(-self._mahalanobis_distance(x_max_zero, means_zero[self.G_max], covs_zero[self.G_max])))
        two = two_left >= two_right # (Equation 21b)
        print(f"two: {two}")
        
        three = torch.det(covs_zero[self.G_min]) >= torch.det(self.covs[self.G_min]) # (Equation 21c)
        print(f"three: {three}")

        four = torch.det(covs_zero[self.G_max]) >= torch.det(self.covs[self.G_max]) # (Equation 21d)
        print(f"four: {four}")

        five_left = torch.sum(torch.exp(-self._mahalanobis_distance(x_min_two, self.means[self.G_min], self.covs[self.G_min])))
        five_left /= torch.sqrt(torch.det(self.covs.data[self.G_min]))
        five_right = torch.sum(torch.exp(-self._mahalanobis_distance(x_max_two, self.means[self.G_max], self.covs[self.G_max])))
        five_right /= torch.sqrt(torch.det(self.covs[self.G_max]))
        five = five_left >= five_right # (Equation 23)
        print(f"five: {five}")
        
        is_convergence = torch.stack([one, two, three, four, five])
        return torch.all(is_convergence).item()
    
    @torch.no_grad()
    def _log_likelihood(
        self,
        x: TensorType["n", "d"], 
        p: TensorType["n", "k"],
        means: TensorType["k", "d"], 
        covs: TensorType["k", "d", "d"]
        ) -> TensorType[""]:
        """Compute log-likelihood of the mixture model in terms of the Mahalanobis.
        
        Args:
            x: Input data of shape [n, d]
            p: Posterior probabilities of shape [n, k].
            means: Component means of shape [k, d].
            covs: Component covariances_ of shape [k, d, d].
        
        Returns:
            torch.Tensor: Log-likelihood value.
        """
        labels = torch.argmax(p, dim=-1) # [n]
        log_2_pi = torch.log(torch.tensor(2 * np.pi, dtype=self.dtype, device=self.device))
        constant_term = -0.5 * self.d * log_2_pi
        
        log_likelihood = torch.tensor(0.0, dtype=self.dtype, device=self.device)
        
        for k in range(self.k): 
            idxs = torch.nonzero(labels == k).view(-1)
            
            if idxs.numel() == 0: 
                continue 
            else: 
                x_k = x[idxs]    
                      
            L = cholesky(covs[k]) # [d, d], covs[k] = L*L^T, L: Lower triangular matrix 
            log_det_cov_k = 2.0 * torch.sum(torch.log(torch.diagonal(L)))     
            
            d_k = self._mahalanobis_distance(x_k, means[k], covs[k], L)
            
            cluster_ll = torch.sum(-0.5 * d_k - 0.5 * log_det_cov_k + constant_term) # mixture weights are omitted deliberately
            log_likelihood += cluster_ll
        
        return log_likelihood / x.shape[0]
    
    @torch.no_grad()
    def _update_gmm(self, x: TensorType["n", "d"]) -> TensorType["bool"]: 
        """Update GMM parameters via iterative entropy-based splitting.
        Args:
            x: Input data of shape [n, d]
        Returns:
            bool: True if converged, False otherwise.
        """
        n, _ = x.shape        
        # Initial posterior update: kmeans hard assignments -> soft assignments 
        p_zero = self._zeroth_posterior_update(x) # [n, k]: posterior probability of each sample belonging to Gaussian component k 
        
        # General computations         
        r_idxs = torch.arange(n, device=self.device) # [n]
        c_idxs = torch.argmax(p_zero, dim=-1) # [n]: Gaussian component k each sample most likely belongs to (hard assignment)
        
        # Initial mixture weights
        counts = torch.zeros(size=(self.k, ), dtype=self.dtype, device=self.device) # [k]
        counts.scatter_add_(0, c_idxs, torch.ones_like(c_idxs, dtype=self.dtype, device=self.device))
        self.weights = counts / n # [k]
        
        # Posterior update loop: 
        for i in tqdm(range(self.max_iter)): 
            # Entropy-related calculation
            G_min, G_max = self._entropy_indices(x, p_zero, r_idxs, c_idxs) # smallest/ biggest entropy, indices of the Gaussians with the smallest/ biggest entropies
            self.G_min = G_min.item() 
            self.G_max = G_max.item() 
            
            self.idxs_min = torch.nonzero(self.G_min == c_idxs).view(-1) # [count_G_min]: indices of all samples belonging to the Gaussian with the smallest entropy
            self.idxs_max = torch.nonzero(self.G_max == c_idxs).view(-1) # [count_G_max]: indices of all samples belonging to the Gaussian with the biggest entropy

            self.count_G_min = self.idxs_min.numel() # []: number of samples in the Gaussian with the smallest entropy 
            self.count_G_max = self.idxs_max.numel() # []: number of samples in the Gaussian with the biggest entropy 
            
            x_min_zero = x.clone()[self.idxs_min] # [count_G_min, d]: samples belonging to the Gaussian with the smallest entropy
            x_max_zero = x.clone()[self.idxs_max] # [count_G_max, d]: samples belonging to the Gaussian with the biggest entropy
            
            means_zero = self.means.clone()
            covs_zero = self.covs.clone()
            
            # First posterior update
            p_one = self._first_posterior_update(x, p_zero, x_min_zero, x_max_zero)

            # Second posterior update
            p_two, x_min_two, x_max_two = self._second_posterior_update(x, p_one)
        
            # Check for convergence
            is_convergence = self._is_convergence(x_min_zero, x_max_zero, x_min_two, x_max_two, means_zero, covs_zero)
            log_likelihood = self._log_likelihood(x, p_two, self.means, self.covs)
            print(log_likelihood)
            
            if self.logger is not None: 
                self.logger.log({"log_likelihood": log_likelihood})
                if  i%10 == 0: 
                    x_cpu = x.clone().cpu().tolist() # copy of data points for later logging
                    feature_cols = [f"dim_{i}" for i in range(self.d)] + ["label"]
                    labels_cpu = torch.argmax(p_two, dim=-1).flatten().detach().cpu().tolist()
                    data = [vec+[lbl] for vec, lbl in zip(x_cpu, labels_cpu)]
                    table = wandb.Table(columns=feature_cols, data=data)
                    self.logger.log({"high_dim_emb": table})
            
            if not is_convergence: 
                p_zero = p_two.clone()  
                # General computations         
                r_idxs = torch.arange(self.n, device=self.device) # [n]
                c_idxs = torch.argmax(p_zero, dim=-1) # [n]: Gaussian component k each sample most likely belongs to (hard assignment)
            else: 
                print(colored("Algorithm has converged!", "green"))
                return None
                              
        return ValueError
    
    def _update_encoder(self, x: TensorType["n", "d"], x_plus: TensorType["n", "d"]) -> TensorType["1, "]: 
        """Compute encoder loss by combining contrastive loss (cl), and bi-directional marginal loss (bml)
        
        Args:
            x: Input data
            x_plus: Positive pairs
        
        Returns:
            loss: Encoder loss
        """
        loss_cl = self._hard_negatives(x, x_plus)
        loss_bml = self._false_negatives(x, x_plus)
        
        loss = loss_cl + self.bml_weight * loss_bml
        
        if self.logger is not None: 
            self.logger.log({"loss_cl": loss_cl})
            self.logger.log({"loss_bml": loss_bml})
            self.logger.log({"loss": loss})
        
        return loss 
    
    def forward(
        self, 
        x: TensorType["n", "d"], 
        x_plus: TensorType["n", "d"], 
        weights: Optional[TensorType["k"]]=None, 
        means: Optional[TensorType["k", "d"]]=None,  
        covs: Optional[TensorType["k", "d", "d"]]=None
        ) -> TensorType[""]:
        """Forward pass: Fit GMM, compute encoder losses
        
        Args:
            x: Input data 
            x_plus: Positive pairs 
            weights: Mixture weights
            means: Component means
            covs: Component covariances

        Returns:
            loss: Encoder loss
        """
        self.dtype = x.dtype
        self.device = x.device
        self.n, d = x.shape
        
        if x.numel() <= 0: raise ValueError("Inputs cannot be empty.") 
        if x_plus.numel() <= 0: raise ValueError("Positive pairs cannot be empty.")
        if d != self.d: raise ValueError(f"The dimension of the data, and the dimension of the model should be equal,  \
            but got dimension {d} for the data, and dimension {self.d} for the model.")
        
        # Initialize GMM parameter 
        if weights is not None: 
            self.weights.copy_(weights)
        if means is not None: 
            self.means.copy_(means)
        if covs is not None: 
            self.covs.copy_(covs)
        
        if not self._is_pos_def(self.covs): 
            self.covs.copy_(self._to_pos_def(self.covs))
        
        # Initialize GMM
        self.component_distribution = D.MultivariateNormal(loc=self.means, covariance_matrix=self.covs)
        is_convergence = self._update_gmm(x.clone().detach()) # pass copy of x, remove that copy from x's computational graph 
        
        if is_convergence :
            self.component_distribution = D.MultivariateNormal(loc=self.means, covariance_matrix=self.covs)
            loss = self._update_encoder(x, x_plus)
            return loss 
        else: 
            return torch.sum(x) * 0.0 + torch.sum(x_plus) * 0.0