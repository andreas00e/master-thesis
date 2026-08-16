from typing import List

import torch 
import torch.nn as nn
import torch.nn.functional as F
from torchtyping import TensorType 


class CrossAttentionQueryPooling(nn.Module): 
    def __init__(
        self,
        k: int,
        d_model: int 
        ) -> None:
        super().__init__()
        
        self.k = k # number of learnable prototypes 
        self.d_model = d_model
        
        q_data = torch.empty(size=(1, 1, self.k, self.d_model), dtype=torch.float32) 
        nn.init.xavier_uniform_(q_data)
        self.q = nn.Parameter(data=q_data) # [1, 1, k, d_model]
        
        self.W_k = nn.Linear(self.d_model, self.d_model, bias=False)
        self.W_v = nn.Linear(self.d_model, self.d_model, bias=False)
        
        self.dropout = nn.Dropout(p=0.1)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: List[TensorType["batch", "n_steps", "d_model"]]): 
        assert len(x) > 0, "At least one condition has to be present!"
        batch_size, n_steps, d_model = x[0].shape
        
        x = torch.stack(tensors=x, dim=2) # [batch_size, n_steps, n_conditions, d_model]: stack conditions 
        
        q = self.q.expand(batch_size, n_steps, -1, -1) # [batch_size, n_steps, k, d_model]
        k = self.W_k(x) # [batch_size, n_steps, n_conditions, d_model]
        v = self.W_v(x) # [batch_size, n_steps, n_conditions, d_model]
        
        attention_weights = q @ k.transpose(-1, -2) # [batch_size, n_steps, k, n_conditions]
        attention_weights = F.softmax(input=(attention_weights / torch.sqrt(torch.tensor(d_model, dtype=torch.float32))), dim=-1) # [batch_size, n_steps, k, n_conditions]: cross attention scores 
        attention_weights = self.dropout(attention_weights)
        
        out = attention_weights @ v # [batch, n_steps, k, d_model] 
        out = self.norm(out)
        
        return out