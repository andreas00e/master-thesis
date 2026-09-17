import math
from typing import Optional

import torch 
import torch.nn as nn 
from torchtyping import TensorType


# !!! ADJUSTED FROM !!!: 
# https://discuss.pytorch.org/t/how-to-modify-the-positional-encoding-in-torch-nn-transformer/104308

class PE(nn.Module): 
    def __init__(self, d_model: int, max_len: int) -> None:
        super().__init__()
        
        if not isinstance(d_model, int) or d_model % 2 != 0: 
            raise ValueError(f"d_model has to be an even integer, got {d_model}.")
        
        self.d_model = d_model 
        self.max_len = max_len 
        
        pe = torch.zeros(self.max_len, self.d_model, dtype=torch.float32) # [max_length, d_model]
        position = torch.arange(0, self.max_len, dtype=torch.float).unsqueeze(1) # [max_len, 1]
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-math.log(10000.0) / self.d_model)) # [d_model / 2]
        pe[:, 0::2] = torch.sin(position * div_term)  
        pe[:, 1::2] = torch.cos(position * div_term) 
        self.register_buffer("pe", pe, persistent=True)
        
    def forward(
        self, 
        x: TensorType["batch*chunk", "1+window", "d_model"], 
        seq_idxs: Optional[TensorType["batch", "chunk", "window"]]=None
        ) -> TensorType["batch*chunk", "1+window", "d_model"]:
                
        if isinstance(seq_idxs, torch.Tensor): 
            pe_idxs = torch.min(seq_idxs, dim=-1, keepdim=True).values # [batch, chunks, 1] 
            pe_idxs = torch.cat(tensors=(pe_idxs, seq_idxs+1), dim=-1) # [batch, chunks, 1+window]
            pe_idxs = pe_idxs.view(-1, x.shape[1]) # [batch*chunk, 1+window]
        else:  
            pe_idxs = torch.arange(x.shape[1], dtype=torch.long, device=x.device) # [1+window]
            pe_idxs = pe_idxs[None, :].expand(x.shape[0], -1) # [batch*chunk, 1+window]
            
        pe_idxs = torch.clamp(pe_idxs, max=self.max_len - 1)
        pe = self.pe[pe_idxs] # [batch*chunk, 1+window, d_model]
    
        return x + pe # [batch*chunk, 1+window, d_model]


class SinusoidalEmbedding(nn.Module): 
    def __init__(self, new_d: int) -> None:
        super().__init__()
        
        if new_d%2 != 0: 
            raise ValueError(f"Expanded dimension has to be even, got {new_d}")
        
        self.new_d = int(new_d)
        half_d = self.new_d // 2 
        
        freq = torch.exp(torch.arange(half_d, dtype=torch.float32) * -(torch.log(torch.tensor(1e4)) / half_d))
        self.register_buffer("freq", freq)        
    
    def forward(self, x: TensorType["n", "1"]) -> TensorType["n", "new_d"]:
        if x.ndim != 2: 
            raise ValueError(f"Expected input to be two-dimensional, got {x.ndim}.") 
        if x.shape[1] != 1: 
            raise ValueError(f"Expected feature dimension of input to be one, got {x.shape[1]}") 
        
        new_x = torch.empty(size=(x.shape[0], self.new_d), dtype=x.dtype, device=x.device) # [n, d]
        scaled_x = x * self.freq
        
        new_x[:, 0::2] = torch.sin(scaled_x) # [n, new_d // 2]
        new_x[:, 1::2] = torch.cos(scaled_x) # [n, new_d // 2]

        return new_x