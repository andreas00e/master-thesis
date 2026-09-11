from typing import List
from omegaconf import DictConfig

import torch 
import torch.nn as nn 
from torchtyping import TensorType


class BottlneckFusion(nn.Module): 
    def __init__(
        self,
        general_kwargs: DictConfig, 
        attn_kwargs: DictConfig
        ) -> None:
        super().__init__()
        
        self.device, self.B, self.d = general_kwargs.values() # B: number of bottleneck tokens 
        self.attn = nn.MultiheadAttention(**attn_kwargs)
        
        self.boto = nn.Parameter(data=torch.empty(size=(self.B, self.d), dtype=torch.float32, device=self.device)) # bottleneck token 
        nn.init.xavier_uniform_(self.boto)
    
    def forward(self, x: List[TensorType["n", "d"]]) -> TensorType["n*m+t", "d"]:
        n, _ = x[0].shape
        m = len(x) # number of modalities 
        
        seq = torch.stack(x).view(-1, self.d) # [n*m, d]
        seq = torch.cat((seq, self.boto))  # [n*m+B, d]
        
        attn_mask = torch.zeros(size=(seq.shape[0], seq.shape[0]), dtype=seq.dtype, device=seq.device) # [n*m+B, n*m+B]
                
        for i in range(m):
              attn_mask[n*i:n*(i+1), :-self.B] = torch.full(size=(n, n*m), fill_value=float("-inf"), dtype=seq.dtype, device=seq.device)
              attn_mask[n*i:n*(i+1), n*i:n*(i+1)] = torch.full(size=(n, n), fill_value=0, dtype=seq.dtype, device=seq.device)
        
        attn_output, _ = self.attn(seq, seq, seq, attn_mask=attn_mask) # [n*m+B, d]
        
        token_output = attn_output[:-self.B, -self.B:] # [n*m, B]
        token_output = torch.stack(torch.chunk(token_output, m)) # [m, ]
        token_output = torch.mean(token_output, dim=1)

        return token_output
         
def main(): 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    n = 2
    B = 2
    d = 4
    m = 2

    model_kwargs = {
        "general_kwargs": {
            "device": device,
            "B": B,
            "d": d   
        },"attn_kwargs": {
            "embed_dim": d, 
            "num_heads": 1, 
            "batch_first": True   
        }
    }
    
    x = torch.randn(size=(n, d), dtype=torch.float32, device=device) # [n, d]
    input = [x.clone() for _ in range(m)] # [n*m, d
    model = BottlneckFusion(**model_kwargs).to(device)
    
    out = model(input)

if __name__ == "__main__": 
    main()