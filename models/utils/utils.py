# ADJUSTED FROM: 
# https://discuss.pytorch.org/t/how-to-modify-the-positional-encoding-in-torch-nn-transformer/104308

import math

import torch 
import torch.nn as nn 
from torchtyping import TensorType


class PE(nn.Module): 
    def __init__(self, d_model: int, max_len: int) -> None:
        super().__init__()
        
        self.d_model = d_model 
        self.max_len = max_len 
        
        self.pe = self._init_pe()
    
    def _init_pe(self): 
        pe = torch.zeros(self.max_len, self.d_model) # [max_length, d_model]
        position = torch.arange(0, self.max_len, dtype=torch.float).unsqueeze(1) # [max_len, 1]
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-math.log(10000.0) / self.d_model)) # [d_model / 2]
        pe[:, 0::2] = torch.sin(position * div_term)  
        pe[:, 1::2] = torch.cos(position * div_term) 
         
        return pe # [max_len, d_model]
    
    def forward(
        self, 
        x: TensorType["batch, chunks, window, d_model"], 
        seq_idxs: TensorType["batch", "chunks", "window"]
        ) -> TensorType["*"]:
        
        idxs_pe = seq_idxs[:, :, :0] # [batch, chunks, 1] 
        idxs_pe = torch.concat(tensors=(idxs_pe, seq_idxs+1), dim=-1) # [batch, chunks, 1+window]
        
        pe = self.pe[idxs_pe.flatten()] # [batch*chunk*(1+window), d_model]
        pe = pe.view(*idxs_pe.shape, -1) # [batch, chunk, 1+window, d_model]
        
        return x + pe


class NormPE(nn.Module):
    """ Normlized Positional Encoding"""
    def __init__(self, d_model: int) -> None:
        super().__init__()
        
        self.projection = nn.Sequential(
            nn.Linear(1, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model)
        )

    def forward(
        self, 
        x: TensorType["batch, chunk, window, d_model"], 
        seq_idxs: TensorType["batch, chunk, window"], 
        seq_len: TensorType["batch"]
        ) -> TensorType["batch, chunk, window, d_model"]:
      
        assert torch.all(seq_len > 1, dim=0), "Every sequence has to contain at least two elements!"
        
        norm_pe = seq_idxs / (seq_len - 1)[: , None, None] # [batch, chunk, window]
        norm_pe = self.projection(norm_pe.unsqueeze(-1)) # [batch, chunk, window, d_model]
        
        return x + norm_pe