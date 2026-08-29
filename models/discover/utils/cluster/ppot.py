import torch 
from torchtyping import TensorType


class PPOT():
    def __init__(
        self, 
        n: int, 
        k: int, 
        device, 
        dtype, 
        T: int,
        eps: float, 
        lota: float, 
        lam: float, 
        rho: float
        ) -> None:
        
        self.T = T
        self.n = n 
        self.k = k 
        self.dtype = dtype 
        self.device = device
        self.eps = eps
        self.rho = rho
        
        self.alpha = torch.full(size=(n, 1), fill_value=1/ n, dtype=self.dtype, device=self.device)
        self.lam = torch.full(size=(self.k+1, 1), fill_value=lam, dtype=self.dtype, device=self.device)
        self.lam[-1] = lota # lota -> "inf" (large value)
        
        self.b = torch.ones(size=(self.k+1, k), dtype=self.dtype, device=self.device)
        self.f = self.lam / (self.lam + self.eps) # [k+1, 1]
        
        self.C = None 
        self.M = None
            
    def forward(self, P: TensorType["n", "k"]): 
        n, k = P.shape
        
        if self.C is None and self.M is None: 
            C_zeros = torch.zeros(size=(n, 1), dtype=P.dtype, device=P.device) # [n, 1]: assignments of virtual cluster 
            self.C = torch.cat(tensors=(-torch.log(P), C_zeros), dim=-1) # [n, k+1]
            self.M = torch.exp(-self.C) / self.eps # [n, k+1]
        
        b = self.b
        for t in range(self.T): 
            rho = self.rho_zero + (1 - self.rho_zero) * torch.exp(-5 * (1 - t / self.T) ** 2)
            beta = torch.full(size=(k+1, 1), fill_value=rho/k, dtype=P.dtype, device=P.device) 
            beta[-1] = 1 - rho
            
            a = self.alpha / (self.M @ self.b) # [n, 1] / ([n, k+1] @ [k+1, 1]) -> [n, 1]
            b_new = ((self.beta / self.M.T @ a) * self.f) ** 2 # ([k+1] / [k+1, n] @ [n, 1]) * [k+1] -> [k+1, 1]
            
            if torch.allclose(b, b_new): 
                break
            else: 
                b = b_new
        
        Q = torch.diag(a) @ self.M @ torch.diag(b) # [n, n] @ [n, k+1] @ [k+1, k+1]
        
        return Q[:, :k] # [n, k+1] -> [n, k]