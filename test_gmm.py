from typing import Dict, Optional, Tuple 

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
        
        self.component_distribution = None 
  
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
        A = (A + A.transpose(-1, -2)) / 2 # make all covariance matrices symmetric

        eigenvals, eigenvecs = eigh(A) # make all covariance matrices positive definite
        eps = torch.abs(torch.amin(eigenvals, dim=-1, keepdim=True)) * 1.1 + 1e-6 # TODO: MOVE THE HYPERPARAMETERS TO THE CONFIG
        eigenvals = torch.where(eigenvals <= 0, eps, eigenvals)
        A = eigenvecs @ torch.diag_embed(eigenvals) @ eigenvecs.transpose(-1, -2)
        
        return A 

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
    
    def _hard_negatives(self, x: TensorType["b", "n", "d"], x_positive: TensorType["b", "n", "d"],
            weights: TensorType["b", "k", ], means: TensorType["b", "k", "d"], covs: TensorType["b", "k", "d", "d"]) -> TensorType["b", "n", "d"]: 
        probs = self._posterior(x, weights, means, covs) # [n, k]: k posterior probabilites
        probs_i_m, probs_i_s, _, idxs_i_s = self._get_m_s(probs)
         
        x_negative = means[idxs_i_s, :] 
        
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
    
    def _false_negatives(self, x: TensorType["b," "n", "d"], x_positive: TensorType["b", "n", "d"]) -> TensorType["b", "n", "d"]:
        n = x.shape[0]
        
        probs = self._posterior(x) # [b, n, k]: k posterior probabilites
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
    
    def _posterior(self, x: TensorType["n", "d"], weights: TensorType["k"], distribution: D.MultivariateNormal) -> TensorType["n", "k"]:
        log_probs = distribution.log_prob(x.unsqueeze(1)) # [n, k]: unweighted log-density
        log_weights = torch.log(weights) # [k]
        log_joint = log_probs + log_weights # [n, k]: log(w_k * log(N(x | mu_k, Sigma_k)) = log(w_k) + log(N(x | mu_k, Sigma_k))
        log_posterior = log_joint - torch.logsumexp(log_joint, dim=-1, keepdim=True) # [n, k]: normalize over k 
        
        return log_posterior.exp() # [n, k]: posterior probabilities p_{i,k} 
    
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
        
        # update weights, means, and covariances
        weights = self.weights.data.clone().detach() 
        weights[G_max] = weight_max
        
        means = self.means.data.clone().detach()
        means[G_max, :] = mean_max
        
        covs = self.covs.data.clone().detach() # [k, d, d]
        covs[G_max, ...] = cov_max 
        
        # get rid of the parameters of the G_min distribution 
        idxs = torch.arange(0, self.k, dtype=torch.int, device=x.device) # [k]
        idxs = idxs[idxs != G_min] # [k-1]
        
        weights = weights[idxs]
        means = means[idxs, :]
        covs = covs[idxs, ...]
        
        component_distribution = self.mixtureModel(means=means, covs=covs)
        
        # recalculate posterior probabilities of the samples belonging to the updated posterior probability
        x_new = torch.vstack((x_min, x_max))
        p_new = self._posterior(x=x_new, weights=weights, distribution=component_distribution) # [count_G_min+count_G_max, k-1]

        return p_new

    def _second_posterior_update(self, x: TensorType["n", "d"], p: TensorType["n", "k"]) -> None: 
        G_min, G_max = self._entropy_indices(x) # [], []: indices of the Gaussians with the smallest and biggest entropies
        cols_idxs = torch.argmax(p, dim=-1) # [n]: Gaussian component k each sample most likely belongs to 
        
        idxs_min = torch.nonzero(G_min == cols_idxs) # [count_G_min]: indices of all samples belonging to the Gaussian with the smallest entropy
        idxs_max = torch.nonzero(G_max == cols_idxs) # [count_G_max]: indices of all samples belonging to the Gaussian with the biggest entropy

        x_min = x[idxs_min] # [count_G_min, d]: samples belonging to the Gaussian with the smallest entropy
        count_G_max = idxs_max.shape[0] # []: number of samples in the Gaussian with the biggest entropy 
        
        weight_max = count_G_max / x.shape[0] # []: new weight of the Gaussian with the biggest entropy 
        self.weights.data[G_max] = weight_max  # update the weight of the Gaussian with the biggest entropy
                    
        mean_max = torch.sum(x_min, dim=0) / count_G_max # [d]: new mean vector of the Gaussian with the biggest entropy 
        self.means.data[G_max] = mean_max # update the mean vector of the Gaussian with the biggest entropy
        
        p_max = p[idxs_max] # [count_G_max, k]: posterior probabilities of all samples belonging to the Gaussian with the biggest entropy 

        num_covs = (p_max - mean_max).unsqueeze(-1) * (p_max - mean_max).unsqueeze(-2) # [n, d, 1] * [n, 1, d] -> [n, d, d]
        cov_max = torch.sum(num_covs) / count_G_max # [d, d]
        self.covs.data[G_max] = cov_max
        
        return None
    
    def _is_convergence(self) -> torch.Tensor.bool:
        return False  

    def _update_gmm(self, x: TensorType["n", "d"])  -> None: 
        # 0th posterior update
        p = self._zeroth_posterior_update(x) # [n, k]: posterior probability of each sample belonging to Gaussian component k 
        
        # general computations         
        r_idxs = torch.arange(x.shape[0], device=x.device) # [n]
        c_idxs = torch.argmax(p, dim=-1) # [n]: Gaussian component k each sample most likely belongs to 
        
        # entropy-related calculation
        E_min, E_max, G_min, G_max = self._entropy_indices(x, p, r_idxs, c_idxs) # [], [], [], []: indices of the Gaussians with the smallest and biggest entropies
        
        idxs_min = torch.nonzero(G_min == c_idxs).squeeze() # [count_G_min]: indices of all samples belonging to the Gaussian with the smallest entropy
        idxs_max = torch.nonzero(G_max == c_idxs).squeeze() # [count_G_max]: indices of all samples belonging to the Gaussian with the biggest entropy
        
        x_min = x[idxs_min] # [count_G_min, d]: samples belonging to the Gaussian with the smallest entropy
        x_max = x[idxs_max] # [count_G_max, d]: samples belonging to the Gaussian with the biggest entropy
        
        count_G_min = idxs_min.shape[0] # []: number of samples in the Gaussian with the smallest entropy 
        count_G_max = idxs_max.shape[0] # []: number of samples in the Gaussian with the biggest entropy 
        
        # posterior update loop: 
        while(self._is_convergence() != True): 
            if E_min <= E_max: 
                # 1st posterior update
                p = self._first_posterior_update(x, p, G_min, G_max, count_G_min, count_G_max, idxs_min, idxs_max, x_min, x_max)
                
                # 2nd posterior update
                p = self._second_posterior_update(x, p)
    
    def _update_encoder(self): 
        pass
        
    def forward(self,  mode: str, x: TensorType["b", "n", "d"], x_plus: Optional[TensorType["b", "n", "d"]]=None) -> TensorType["*"]:
        self.component_distribution = self.mixtureModel(means=self.means, covs=self.covs)
        
        if mode == "update_gmm": 
            return self._update_gmm(x) 

        elif mode == "update_encoder": 
            return self._update_encoder(x, x_plus) 
        
def main(): 
    d = 128
    
    kwargs = {
        "k": 20, 
        "d": d,
        "alpha": 0.1, 
        "beta": 0.2, 
        "sim": {
            "dim": 1.0, 
            "eps": 1.0e-6
        }
    } 
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    
    gmm = GMM(**kwargs).to(device)
    x = torch.rand(size=(100, d)).to(device) * 2 - 1
    
    out = gmm(mode="update_gmm", x=x)
    print("Hello")

if __name__ == "__main__": 
    main()