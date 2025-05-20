import torch 
import torch.nn as nn 
from termcolor import colored

class EmbeddingCombiner(nn.Module):
    def __init__(self, batch_size: int, hidden_size:int):
        super().__init__()       
        # Learnable temperature  
        self.T = nn.Parameter(data=torch.ones(size=(batch_size, hidden_size)), requires_grad=True)
    
    def forward(self, emb1, emb2):
        assert emb1.shape == emb2.shape, colored("Dimensions of given tensors do not match", 'red')
        d = torch.tensor(emb1.shape[-1]) # Dimenison of input for normalization
        z = emb1 * emb2 / torch.sqrt(d) 
        logit = torch.exp(z / self.T) 
        out = logit / torch.sum(logit, dim=-1, keepdim=True)
        return out, self.T
