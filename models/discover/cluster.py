from typing import Dict, Optional, Tuple 

import torch
import torch.nn as nn 
import torch.nn.functional as F
import torch.distributions as D
from torchtyping import TensorType
from torch.linalg import cholesky_ex, eigh


class KMeans(nn.Module):
    def __init__(
        self, 
        k: int=3, 
        max_iter: int=300, 
        tol: float=1e-4
        ):
        super().__init__()
        
        self.k = k
        self.max_iter = max_iter
        self.tol = tol
        self.register_buffer("means", torch.empty(0)) # [k, d]
        self.is_fitted = False
         
    def _compute_distances(self, x: TensorType["n", "d"]) -> TensorType["n", "k"]:
        # ||X - C||^2 = ||X||^2 + ||C||^2 - 2*X*C.T  
        x_norm = (x ** 2).sum(dim=1, keepdim=True) # [n, 1]
        c_norm = (self.means ** 2).sum(dim=1, keepdim=True).T # [1, k]
        distances = x_norm + c_norm- 2 * x @ self.means.T # [n, k]
        return distances
     
    def _fit(self, x: TensorType["n", "d"]) -> "KMeans":
        n, d = x.shape
        assert n >= self.k, "Need at least k points"
        
        # Random init from data 
        rand_idxs = torch.randperm(n, device=x.device)[:self.k]
        self.means.resize_(self.k, d).copy_(x[rand_idxs])
        
        for _ in range(self.max_iter):
            old_means = self.means.data.clone() #
            distances = self._compute_distances(x) # [n, n_clusters]:
            labels = torch.argmin(distances, dim=1) # [n]: hard cluster assignments
            one_hot = F.one_hot(labels, num_classes=self.k).to(x.dtype) # [n, k]: one-hot centroid mapping matrix       
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

        weights = counts.float() / x.shape[0] # [k], sum to 1 
        
        covs = self._get_covariance(x, labels) # [k, d, d]
        
        return {
            "weights": weights, # [k]
            "means": self.means.data, #[k, d]
            "covs": covs, # [k, d, d]
            "labels": labels # [n]
        } 

class GMM(nn.Module): 
    def __init__(
        self, 
        k: int, # number of Gaussian distributions  
        d: int, # dimension of data and parameters of Gaussian distributions 
        alpha: float, 
        beta: float, 
        sim: Dict, # parameters of cosine similarity
        weights: Optional[TensorType["k", "d"]] = None, # mixture weights of GMM's individual distributions 
        means: Optional[TensorType["k", "d"]] = None, # mean vectors of Gaussian distributions 
        covs: Optional[TensorType["k", "d", "d"]] = None # covariance matrices of Gaussian distributions 
        ) -> None: 
        
        super().__init__()
        
        self.k = k 
        self.d = d
        self.alpha = alpha 
        self.beta = beta 
        self.sim = sim
        
        self.weights = weights if isinstance(weights, nn.Parameter) else nn.Parameter(weights) if weights is not None else nn.Parameter(data=torch.ones(size=(self.k, )) / self.k) # [k]
        self.means = means if isinstance(means, nn.Parameter) else nn.Parameter(means) if means is not None else nn.Parameter(data=torch.randn(size=(self.k, self.d))) # [k, d]
        self.covs = covs if isinstance(covs, nn.Parameter) else nn.Parameter(covs) if covs is not None else nn.Parameter(data=torch.eye(self.d).unsqueeze(0).repeat(self.k, 1, 1)) # [k, d, d]
        self.tau = nn.Parameter(data=torch.ones((1, ), dtype=torch.float64)) # [1]
        self.lam = nn.Parameter(data=torch.ones((1, ), dtype=torch.float64)) # [1]
        
        self.component_distribution, self.dtype, self.device = [None]*3
  
        self.sim = nn.CosineSimilarity(**self.sim)
        self.relu = nn.ReLU()
        
    def mixtureModel(self, means: TensorType["k", "d"], covs: TensorType["k", "d", "d"]) -> D.MultivariateNormal:        
        if not self.is_pos_def(covs):
            covs = self.to_pos_def(covs)
        
        component_distribution = D.MultivariateNormal(loc=means, covariance_matrix=covs)
        
        return component_distribution
    
    def is_pos_def(self, A: TensorType["k", "d", "d"]) -> bool:
        if not torch.allclose(A, A.transpose(-1, -2)): # ensure that the matrix is symmetrix first  
            return False 
        
        _, info = cholesky_ex(A) # Cholesky decomposition

        return bool(torch.all(info == 0))
    
    def to_pos_def(self, A: TensorType["k", "d", "d"]) -> TensorType["k", "d", "d"]:
        A = (A + A.transpose(-1, -2)) / 2  # make all covariance matrices symmetric

        # make all covariance matrices positive definite
        eigenvals, _ = eigh(A)
        min_eig = eigenvals.amin(dim=-1)  

        shift = torch.clamp(-min_eig, min=0) * 1.1 + 1e-6  # TODO: MOVE THE HYPERPARAMETERS TO THE CONFIG
        eye = torch.eye(A.shape[-1], device=A.device, dtype=A.dtype)
        A = A + shift.view(-1, 1, 1) * eye

        return A

    def _posterior(self, x: TensorType["n", "d"], weights: TensorType["k"], distribution: D.MultivariateNormal) -> TensorType["n", "k"]:
        log_probs = distribution.log_prob(x.unsqueeze(1)) # [n, k]: unweighted log-density
        log_weights = torch.log(weights) # [k]
        log_joint = log_probs + log_weights # [n, k]: log(w_k * log(N(x | mu_k, Sigma_k)) = log(w_k) + log(N(x | mu_k, Sigma_k))
        log_posterior = log_joint - torch.logsumexp(log_joint, dim=-1, keepdim=True) # [n, k]: normalize over k 
        
        return log_posterior.exp() # [n, k]: posterior probabilities p_{i,k} 
    
    def confidence(self, x: TensorType["n", "d"]) -> TensorType["n"]: 
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
        
        loss_bml = weight / weight_avg * (self.relu(delta + self.alpha) + (self.relu(-delta - self.beta)))
        loss_bml = torch.scatter_add(output_tensor, 0, inverse_indices, loss_bml)
        
        if unique_elements.shape != n: 
            out = torch.zeros(size=(n, ), device=x.device)
            out[unique_elements] = loss_bml
            loss_bml = out 

        return loss_bml
    
    def _entropy_indices(self, x: TensorType["n", "d"], p: TensorType["n", "k"], r_idxs: TensorType["n"], c_idxs: TensorType["n"]) -> Tuple[TensorType["1"], ...]:
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
    
    def _zeroth_posterior_update(self, x: TensorType["n", "d"]) -> TensorType["n", "k"]:
        return self._posterior(x, self.weights, self.component_distribution) # [n, k]
        
    def _first_posterior_update(self, x, p, G_min, G_max, count_G_min, count_G_max, idxs_min, idxs_max, x_min, x_max) -> TensorType["count_G_max", "k"]: 
        # update the mixture weight of the Gaussian with the biggest entropy 
        weight_max = (count_G_max + count_G_min) / x.shape[0] # []: new mixture weight of the Gaussian with the biggest entropy 
        
        # update the mean vector of the Gaussian with the biggest entropy
        mean_max = (torch.sum(x_min, dim=0) + torch.sum(x_max, dim=0)) / (count_G_max + count_G_min) # [d]: new mean vector of the Gaussian with the biggest entropy 
        
        # exchange of posterior probabilities 
        p_min = p[idxs_min].clone()  # [count_G_min, k]
        p_min[:, [G_min, G_max]] = p_min[:, [G_max, G_min]]
        p[idxs_min] = p_min  # [n, k]
        
        # update of the covariance matrix of the Gaussian with the biggest entropy
        numerator_covs = p[:, G_max][:, None, None] * ((x - mean_max).unsqueeze(-1) * (x - mean_max).unsqueeze(-2)) # [n, 1, 1] * ([n, d, 1] * .[n, 1, d]) -> [n, d, d]
        numerator_covs = torch.sum(numerator_covs, dim=0) # [d, d]
        cov_max = numerator_covs / (count_G_max + count_G_min) # [d, d]: new covariance matrix of the Gaussian with the biggest entropy 
        
        # update weight
        weights = self.weights.data.clone().detach() 
        weights[G_max] = weight_max # [k]
        
        # update means
        means = self.means.data.clone().detach()
        means[G_max, :] = mean_max # [k, d]
        
        # update covariance matrices
        covs = self.covs.data.clone().detach() 
        covs[G_max, ...] = cov_max # [k, d, d]
        
        # get rid of the parameters of the G_min distribution 
        idxs = torch.arange(0, self.k, dtype=torch.int, device=x.device) # [k]
        idxs = idxs[idxs != G_min] # [k-1]
        
        weights = weights[idxs] # [k-1]
        means = means[idxs, :] # [k-1, d]
        covs = covs[idxs, ...] # [k-1, d, d]
        
        component_distribution = self.mixtureModel(means=means, covs=covs)
        
        # recalculate posterior probabilities of the samples belonging to the updated posterior probability
        x_new = torch.vstack((x_min, x_max)) # [count_G_min+count_G_max, d]
        p_new = self._posterior(x=x_new, weights=weights, distribution=component_distribution) # [count_G_min+count_G_max, k-1]

        return p_new, weight_max, mean_max, cov_max

    def _second_posterior_update(self, x, p_0, p_1, weight_max, mean_max, cov_max, G_min, G_max, count_G_min, count_G_max, idxs_min, idxs_max, x_min, x_max) -> None:         
        x_new = torch.vstack(tensors=(x_min, x_max)) # [count_G_min+count_G_max, d]: samples belonging to the merged Gaussian distribution 
        idx_max = torch.argmax(torch.diagonal(cov_max)) # []: dimension with the biggest variance
        
        # swap of posterior probabilities 
        idxs_swap = x_new[:, idx_max] <= mean_max[idx_max]
        r_swap = torch.nonzero(idxs_swap).squeeze()
        p_1[r_swap][:, [G_min, G_max]] = p_1[r_swap][:, [G_max, G_min]] # [count_G_min+count_G_max, k-1]
        
        idxs = torch.hstack((idxs_min, idxs_max)) # [count_G_min+count_G_max]  
        
        p_max = p_0[:, G_max].clone() 
        p_max[idxs] = p_1[:, G_max]
        
        # update the mixture weights of the two new Gaussian distributions
        weight_min = torch.tensor(count_G_min / x.shape[0]) # []: new weight of the Gaussian with the past smallest entropy 
        weight_max = torch.tensor(count_G_max / x.shape[0]) # []: new weight of the Gaussian with the past biggest entropy 
        weights_two = torch.hstack((weight_min, weight_max)).to(self.dtype).to(self.device) # [2]
        
        # update the mean vectorss of the two new Gaussian distributions
        mean_min = torch.sum(x_min, dim=0) / count_G_min # [d]: new mean vector of the Gaussian with the biggest entropy                
        mean_max = torch.sum(x_max, dim=0) / count_G_max # [d]: new mean vector of the Gaussian with the past biggest entropy 
        means_two = torch.vstack((mean_min, mean_max)) # [2, d]
        
        cov_min = p_max[:, None, None] * ((x - mean_min).unsqueeze(-1) * (x - mean_min).unsqueeze(-2)) # [n, d, 1] * [n, 1, d] -> [n, d, d]
        cov_min = torch.sum(cov_min, dim=0) / count_G_min # [d, d]
        cov_max = p_max[:, None, None] * ((x - mean_max).unsqueeze(-1) * (x - mean_max).unsqueeze(-2)) # [n, d, 1] * [n, 1, d] -> [n, d, d]
        cov_max = torch.sum(cov_max, dim=0) / count_G_max # [d, d]
        covs_two = torch.vstack((cov_min[None, ...], cov_max[None, ...]))
        
        # update weights
        weights = self.weights.data.clone() 
        weights[[G_min.item(), G_max.item()]] = weights_two
        
        # update means 
        means = self.means.data.clone()
        means[[G_min.item(), G_max.item()], :] = means_two
        
        # update covariance matrices 
        covs = self.covs.data.clone() 
        covs[[G_min.item(), G_max.item()], ...] = covs_two 
        
        component_distribution = self.mixtureModel(means=means, covs=covs)
        p_2 = self._posterior(x=x_new, weights=weights, distribution=component_distribution)

        return p_2
    
    def _is_convergence(self) -> torch.Tensor.bool:
        return False  

    def _update_gmm(self, x: TensorType["n", "d"])  -> None: 
        # 0th posterior update
        p_0 = self._zeroth_posterior_update(x) # [n, k]: posterior probability of each sample belonging to Gaussian component k 
        
        # general computations         
        r_idxs = torch.arange(x.shape[0], device=x.device) # [n]
        c_idxs = torch.argmax(p_0, dim=-1) # [n]: Gaussian component k each sample most likely belongs to 
        
        # entropy-related calculation
        E_min, E_max, G_min, G_max = self._entropy_indices(x, p_0, r_idxs, c_idxs) # [], [], [], []: indices of the Gaussians with the smallest and biggest entropies
        
        idxs_min = torch.nonzero(G_min == c_idxs).squeeze() # [count_G_min]: indices of all samples belonging to the Gaussian with the smallest entropy
        idxs_max = torch.nonzero(G_max == c_idxs).squeeze() # [count_G_max]: indices of all samples belonging to the Gaussian with the biggest entropy
        
        x_min = x[idxs_min] # [count_G_min, d]: samples belonging to the Gaussian with the smallest entropy
        x_max = x[idxs_max] # [count_G_max, d]: samples belonging to the Gaussian with the biggest entropy
        
        count_G_min = idxs_min.numel() # []: number of samples in the Gaussian with the smallest entropy 
        count_G_max = idxs_max.numel() # []: number of samples in the Gaussian with the biggest entropy 
        
        # posterior update loop: 
        while(self._is_convergence() != True): 
            if E_min <= E_max: 
                # 1st posterior update
                p_1, weight_1, mean_1, cov_2 = self._first_posterior_update(x, p_0, G_min, G_max, count_G_min, count_G_max, idxs_min, idxs_max, x_min, x_max)
                # 2nd posterior update
                p_2 = self._second_posterior_update(x, p_0, p_1, weight_1, mean_1, cov_2, G_min, G_max, count_G_min, count_G_max, idxs_min, idxs_max, x_min, x_max)
    
    def _update_encoder(self, x, x_plus):
        loss_cl = self._hard_negatives(x, x_plus)
        loss_bml = self._false_negatives(x, x_plus)
        return loss_cl + loss_bml
        
    def forward(self,  mode: str, x: TensorType["b", "n", "d"], x_plus: Optional[TensorType["b", "n", "d"]]=None) -> TensorType["*"]:
        self.dtype = x.dtype
        self.device = x.device
        self.component_distribution = self.mixtureModel(means=self.means, covs=self.covs)
        
        if mode == "update_gmm": 
            return self._update_gmm(x) 

        elif mode == "update_encoder": 
            return self._update_encoder(x, x_plus) 