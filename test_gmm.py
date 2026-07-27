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
        self.means = means if isinstance(means, nn.Parameter) else nn.Parameter(means) if means is not None else nn.Parameter(data=torch.rand(size=(self.k, self.d))) # [k, d]
        self.covs = covs if isinstance(covs, nn.Parameter) else nn.Parameter(covs) if covs is not None else nn.Parameter(data=torch.rand(size=(self.k, self.d, self.d))) # [k, d, d]
        self.tau = nn.Parameter(data=torch.ones(1, )) # [1]
        self.lam = nn.Parameter(data=torch.ones(1, )) # [1]
        
        self.component_distribution, self.mixture = None, None 
  
        self.sim = nn.CosineSimilarity(**self.sim)
        self.relu = nn.ReLU()
        
    def mixtureModel(self) -> Tuple[D.MultivariateNormal, D.MixtureSameFamily]:
        """
        Create a mixture of Gaussian distributions.
        
        Constructs a MixtureSameFamily distribution from the learned weights, means,
        and covariance matrices. Ensures covariance matrices are positive definite.
        
        Returns:
            Tuple[D.MultivariateNormal, D.MixtureSameFamily]: Component distribution and mixture distribution.
        """
        mixture_distribution = D.Categorical(probs=self.weights.data) 
        
        if not self.is_pos_definite(self.covs.data):
            self.covs.data = self.to_pos_definite(self.covs.data)
        
        component_distribution = D.MultivariateNormal(loc=self.means.data, covariance_matrix=self.covs.data)
        mixture = D.MixtureSameFamily(mixture_distribution=mixture_distribution, component_distribution=component_distribution)
        
        return component_distribution, mixture
    
    def is_pos_definite(self, A: TensorType["k", "d", "d"]) -> bool:
        """
        Check if a batch of matrices is positive definite.
        
        Verifies symmetry first, then performs Cholesky decomposition to determine
        positive definiteness.
        
        Args:
            A: Batch of square matrices of shape [k, d, d].
            
        Returns:
            bool: True if all matrices are positive definite, False otherwise.
        """
        
        if not torch.allclose(A, A.transpose(-1, -2)): # ensure that the matrix is symmetrix first  
            return False 
        
        _, info = cholesky_ex(A) # Cholesky decomposition

        return bool(torch.all(info == 0))
   
    def to_pos_definite(self, A: TensorType["k", "d", "d"]) -> TensorType["k", "d", "d"]:
        """
        Convert matrices to positive definite form.
        
        Symmetrizes matrices and adjusts eigenvalues to ensure positive definiteness
        by replacing non-positive eigenvalues with a small positive threshold.
        
        Args:
            A: Batch of matrices of shape [k, d, d].
            
        Returns:
            TensorType["k", "d", "d"]: Positive definite matrices.
        """
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
    
    def _entropy_counts(self, x: TensorType["n", "d"]) -> Tuple[TensorType["1"], TensorType["1"]]:
        """
        Compute entropy counts for each Gaussian component and identify extreme components.

        Args:
            x: Input tensor of shape [n, d].

        Returns:
            Tuple with indices of the Gaussian component with minimum aggregated entropy
            and the component with maximum aggregated entropy.
        """
        probs = self._posterior_unweighted(x) # [n, k]: likelihood of each sample belonging to Gaussian component k 
        
        rows_idxs = torch.arange(x.shape[0], device=x.device) # [n]
        cols_idxs = torch.argmax(probs, dim=-1) # [n]: Gaussian component k each sample most likely belongs to 

        probs = probs[rows_idxs, cols_idxs]# [n]: biggest likelihood for each sample       
        probs *= torch.log(probs) # [n]: entropy for each sample 
         
        unique_elements, inverse_indices = torch.unique(cols_idxs, return_inverse=True) # [n], [n]
        output_tensor = torch.zeros(size=(unique_elements.shape[0], ), device=x.device) # [n']
        
        E_k = torch.zeros(size=(self.k, ), device=x.device) # [k]
        E_k[unique_elements] = -torch.scatter_add(output_tensor, 0, inverse_indices, probs) # [k]
        
        G_min = torch.argmin(E_k) # [1]: distribution with the largest entropy
        G_max = torch.argmax(E_k) # [1]: distribution with the smallest entropy
        
        return G_min, G_max
        
    def _posterior_unweighted(self, x: TensorType["n", "d"]) -> TensorType["n", "k"]:
        """
        Compute unweighted posterior probabilities for the current GMM components.

        Args:
            x: Input tensor of shape [n, d].

        Returns:
            Tensor of shape [n, k] containing posterior probabilities p_{i,k}.
        """
        log_probs = self.component_distribution.log_prob(x.unsqueeze(1)) # [n, k]: unweighted log-density
        log_weights = torch.log(self.weights.data) # [k]
        log_joint = log_probs + log_weights[None, :].expand(x.shape[0], -1)  # [n, k]: log(w_k * log(N(x | mu_k, Sigma_k)) = log(w_k) + log(N(x | mu_k, Sigma_k))
        log_posterior = log_joint - torch.logsumexp(log_joint, dim=-1, keepdim=True)  # normalize over k 
        
        return log_posterior.exp()  # [n, k]: posterior probabilities p_{i,k}

    def _posterior_weighted(self, x: TensorType["n", "d"]) -> TensorType["n"]:
        """
        Compute the mixture density for each sample under the current GMM.

        Args:
            x: Input tensor of shape [n, d].

        Returns:
            Tensor of shape [n] containing the mixture density for each sample.
        """
        return self.mixture.log_prob(x).exp() # [n]: posterior probability of each sample  
    
    def _zeroth_posterior_update(self, x: TensorType["n", "d"]) -> TensorType["n", "k"]:
        """
        Perform the zeroth posterior update step using the current mixture density.

        Args:
            x: Input tensor of shape [n, d].

        Returns:
            Tensor of shape [n] containing the current weighted posterior approximation.
        """
        return self._posterior_weighted(x) # [n, k]
        
    def _first_posterior_update(self, x: TensorType["n", "d"], p: TensorType["n", "k"]) -> TensorType["count_G_max", "k"]: 
        """
        Update the GMM using samples from entropy-ranked components and compute a new posterior.

        Args:
            x: Input tensor of shape [n, d].
            p: Posterior probability tensor of shape [n, k].

        Returns:
            Tensor containing updated posterior probabilities for the selected samples.
        """
        G_min, G_max = self._entropy_counts(x) # [], []: indices of the Gaussians with the smallest and biggest entropies
        cols_idxs = torch.argmax(p, dim=-1) # [n]: Gaussian component k each sample most likely belongs to 
        
        idxs_min = torch.nonzero(G_min == cols_idxs) # [count_G_min]: indices of all samples belonging to the Gaussian with the smallest entropy
        idxs_max = torch.nonzero(G_max == cols_idxs) # [count_G_max]: indices of all samples belonging to the Gaussian with the biggest entropy
        
        x_min = x[idxs_min] # [count_G_min, d]: samples belonging to the Gaussian with the smallest entropy
        x_max = x[idxs_max] # [count_G_max, d]: samples belonging to the Gaussian with the biggest entropy
        
        count_G_min = idxs_min.shape[0] # []: number of samples in the Gaussian with the smallest entropy 
        count_G_max = idxs_max.shape[0] # []: number of samples in the Gaussian with the biggest entropy 
        
        weight_max = (count_G_max+count_G_min) / (x.shape[0]-1) # []: new weight of the Gaussian with the biggest entropy 
        self.weights.data[G_max] = weight_max # update the weight of the Gaussian with the biggest entropy 
        
        mean_max = torch.sum(x_min+x_max, dim=0) / (count_G_max+count_G_min) # [d]: new mean vector of the Gaussian with the biggest entropy 
        self.means.data[G_max] = mean_max # update the mean vector of the Gaussian with the biggest entropy
        
        p_max = p[idxs_max] # [count_G_max, k]: posterior probabilities of all samples belonging to the Gaussian with the biggest entropy 
        
        numerator_covs = (p_max-mean_max).unsqueeze(-1) * (p_max-mean_max).unsqueeze(-2) # [n, d, 1] * [n, 1, d] -> [n, d, d]
        numerator_covs = torch.sum(numerator_covs) # [d, d]
        cov_max = numerator_covs / (count_G_max+count_G_min) # [d, d]: : new covariance matrix of the Gaussian with the biggest entropy 
        self.covs.data[G_max] = cov_max # update the covatiance matrix of the Gaussian with the biggest entropy
        
        self.component_distribution, self.mixture = self.mixtureModel() # update the GMM
        
        x_1 = torch.vstack((x_min, x_max)) # [count_G_min + count_G_max, d]
        p_1_unweighted = self._posterior_unweighted(x_1) # [count_G_min + count_G_max, k]
        
        n = weight_max * p_1_unweighted # [count_G_min + count_G_max, k]
        d = torch.sum(n, dim=-1) # [count_G_min + count_G_max]
        p_new_weighted = n / d  # [count_G_min + count_G_max, k]
        
        return p_new_weighted 

    def _second_posterior_update(self, x: TensorType["n", "d"], p: TensorType["n", "k"]) -> None: 
        p = self._posterior_weighted(x) # [n, k]: k posterior probabilities
        G_min, G_max = self._entropy_counts(x) # [1], [1]: indices of the distributions with the smallest and biggest entropies

        cols_idxs = torch.argmax(p, dim=-1) # [n]
        
        idxs_min = torch.nonzero(G_min == cols_idxs) 
        idxs_max = torch.nonzero(G_max == cols_idxs) 
        
        count_G_max = idxs_max.shape[0]
        
        weight_max = count_G_max / x.shape[0]
        self.weights.data[G_max] = weight_max # update weight of distribution with maximum entropy 

        x_min = x[idxs_min]
                
        mean_max = torch.sum(x_min, dim=0) / count_G_max
        self.means.data[G_max] = mean_max # update mean vector of distribution with maximum entropy 
        
        p_x_max = p[idxs_max] # [count_g_max, self.k]

        num_covs = (p_x_max - mean_max).unsqueeze(-1) * (p_x_max - mean_max).unsqueeze(-2) # [n, d, 1] * [n, 1, d] -> [n, d, d]
        cov_max = torch.sum(num_covs) / count_G_max # [d, d]
        self.covs.data[G_max] = cov_max
        
        return None
    
    def _convergion_crieria(self):
        pass 

    def _update_gmm(self, x: TensorType["n", "d"])  -> None: 
        p_0 = self._zeroth_posterior_update(x)
        p_1 = self._first_posterior_update(x, p_0)
        p_2 = self._second_posterior_update(x, p_1)
    
    def _update_encoder(self): 
        pass
        
    def forward(self,  mode: str, x: TensorType["b", "n", "d"], x_plus: Optional[TensorType["b", "n", "d"]]=None) -> TensorType["*"]:
        self.component_distribution, self.mixture = self.mixtureModel()
        
        if mode == "update_gmm": 
            return self._update_gmm(x) 

        elif mode == "update_encoder": 
            return self._update_encoder(x, x_plus) 
        
def main(): 
    kwargs = {
        "k": 20, 
        "d": 512,
        "alpha": 0.1, 
        "beta": 0.2, 
        "sim": {
            "dim": 1.0, 
            "eps": 1.0e-6
        }
    } 
    
    gmm = GMM(**kwargs)
    x = torch.rand(size=(100, 512))    
    
    out = gmm(mode="update_gmm", x=x)

if __name__ == "__main__": 
    main()