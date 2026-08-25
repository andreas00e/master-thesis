import math 

import torch 
import torch.nn as nn 

from torchtyping import TensorType


class PPOT(nn.Module): 
    def __init__(
        self, 
        device: str,  
        N: int, 
        K: int, 
        T: int, 
        eps: float,
        iota: float, 
        lam: float, 
        rho_zero: float 
        ) -> None:
        super().__init__()
        
        self.device = torch.device(device)
        self.T = T
        self.eps = eps
        self.rho_zero = rho_zero
                
        self.register_buffer("alpha", torch.full(size=(N, 1), fill_value=1/N, dtype=torch.float32, device=self.device))
                        
        lam = torch.full(size=(K+1, 1), fill_value=lam, dtype=torch.float32, device=self.device)
        lam[-1] = iota 
        self.register_buffer("f", lam / (lam + eps))      
        
        self.register_buffer("b", torch.ones(size=(K+1, 1), dtype=torch.float32, device=self.device))
  
    def forward(self, P: TensorType["n", "k"], rho: float=None) -> TensorType["n, k"]: 
        n, k = P.shape
        zeros_c = torch.zeros(size=(n, 1), dtype=P.dtype, device=P.device) # [n, 1]
        C = torch.cat((-torch.log(P+1e-12), zeros_c), dim=-1) # [n, k+1]: 1e-12 -> stability offset
        M = torch.exp(-C / self.eps) # [n, k+1]
        
        rho = rho if rho is not None else self.rho_zero
        beta = torch.full((k + 1, 1), rho / k, dtype=torch.float32, device=P.device)
        beta[-1] = 1.0 - rho

        b = self.b.clone()
        for _ in range(self.T): 
            b_old = b.clone() 
            
            a = self.alpha / (M @ b_old) # [n, 1] / [n, k+1] @ [k+1, 1] -> [n, 1]
            b = torch.pow((beta / (M.T @ a)), self.f) # [k+1, 1] / [k+1, n] @ [n, 1] -> [k+1, 1]
            
            if torch.allclose(b_old, b, rtol=1e-6): 
                break
        
        self.b.copy_(b)
                    
        Q = a * M * b.T # [n, n] @ [n, k+1] @ [k+1, k+1] -> [n, k+1]
        return Q[:, :-1] # [n, k]
    

def main(): 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    N = 10
    K = 2
    
    model_kwargs = {
        "device": device,
        "N": N, 
        "K": K, 
        "eps": 0.1, 
        "iota": float("inf"), 
        "lam": 1.0, 
        "rho_zero": 0.1, 
    }
    
    P = torch.randn(size=(N, K), dtype=torch.float32, device=device)
    model = PPOT(**model_kwargs).to(device)
    
    out = model(P)
    

if __name__ == "__main__": 
    main() 