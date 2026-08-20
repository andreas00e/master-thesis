# ADJUSTED FROM: 
# https://discuss.pytorch.org/t/how-to-modify-the-positional-encoding-in-torch-nn-transformer/104308

import math
from typing import Optional

import torch 
import torch.nn as nn 
from torchtyping import TensorType

class PE(nn.Module): 
    def __init__(self, d_model: int, max_len: int) -> None:
        super().__init__()
        
        assert isinstance(d_model, int) and (d_model % 2 == 0), "d_model has to be an even integer!"
        self.d_model = d_model 
        self.max_len = max_len 
        
        pe = torch.zeros(self.max_len, self.d_model, dtype=torch.float32) # [max_length, d_model]
        position = torch.arange(0, self.max_len, dtype=torch.float).unsqueeze(1) # [max_len, 1]
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-math.log(10000.0) / self.d_model)) # [d_model / 2]
        pe[:, 0::2] = torch.sin(position * div_term)  
        pe[:, 1::2] = torch.cos(position * div_term) 
        self.register_buffer("pe", pe)
        
    def forward(
        self, 
        x: TensorType["...", "d_model"], 
        seq_idxs: Optional[TensorType["batch", "chunks", "window"]]=None # positions of elemetns of x in its original sequence 
        ) -> TensorType["batch*chunk", "1+window", "d_model"]:
                        
        if isinstance(seq_idxs, torch.Tensor): 
            pe_idxs = torch.min(seq_idxs, dim=-1, keepdim=True).values # [batch, chunks, 1] 
            pe_idxs = torch.cat(tensors=(pe_idxs, seq_idxs+1), dim=-1) # [batch, chunks, 1+window]
            pe = self.pe[pe_idxs]
            pe = pe.view(*x.shape)
            
        elif seq_idxs is None: 
            pe = self.peI
            
        else: 
            raise ValueError

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