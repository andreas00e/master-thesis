import torch 
import torch.nn as nn 
from torchtyping import TensorType 
 
# ADAPTED FROM: https://github.com/real-stanford/xskill/blob/main/xskill/model/core.py
class Sinkhorn(nn.Module): 
    def __init__(
        self, 
        epsilon: float, 
        num_iterations: int
        ) -> None: 
        super().__init__()    
    
        
        self.register_buffer("epsilon", torch.tensor(epsilon, dtype=torch.float32)) 
        self.register_buffer("num_iterations", torch.tensor(num_iterations, dtype=torch.long))
                    
    @torch.no_grad()
    def forward(self, out: TensorType["n", "k"]) -> TensorType["n", "k"]:
        Q = torch.exp(out / self.epsilon).T # [K, B]
        K, B = Q.shape # number of prototypes, number of samples to assign

        sum_Q = torch.sum(Q) + 1e-8 # matrix has to sum to 1

        Q /= sum_Q

        for _ in range(self.num_iterations):
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True) + 1e-8 # [K, 1]: normalize each row: total weight per prototype must be 1/K

            Q /= sum_of_rows 
            Q /= K

            Q /= torch.sum(Q, dim=0, keepdim=True) + 1e-8 # [1, B]: normalize each column: total weight per sample must be 1/B
            Q /= B

        Q *= B  # the colomns must sum to 1 so that Q is an assignment
        Q = Q.T 
            
        return Q