from typing import List

import torch 
import torch.nn as nn
import torch.nn.functional as F
from torchtyping import TensorType 


class CrossAttentionQueryPooling(nn.Module): 
    def __init__(
        self,
        d_model: int, 
        m: int=4
        ) -> None:
        super().__init__()
        
        self.d_model = d_model
        self.m = m # number of learnable prototypes 
        
        self.q = nn.Parameter(torch.empty(size=(1, 1, 1, self.m, self.d_model), dtype=torch.float32)) # [1, 1, k, d_model]
        nn.init.xavier_uniform_(self.q)

        self.W_k = nn.Linear(self.d_model, self.d_model, bias=False)
        self.W_v = nn.Linear(self.d_model, self.d_model, bias=False)
        
        self.dropout = nn.Dropout(p=0.1)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: List[TensorType["batch", "n_steps", "condition_horizon" "d_model"]]) -> TensorType["batch", "n_steps", "k", "d_model"]: 
        assert len(x) > 0, "At least one condition has to be present!"
        batch_size, n_steps, condition_horizon, d_model = x[0].shape
        
        x = torch.stack(tensors=x, dim=-2) # [batch_size, n_steps, condition_horizon n_conditions, d_model]: stack conditions 
        
        q = self.q.expand(batch_size, n_steps, condition_horizon, -1, -1) # [batch_size, n_steps, condition_horizon, m, d_model]
        k = self.W_k(x) # [batch_size, n_steps, condition_horizon, n_conditions, d_model]
        v = self.W_v(x) # [batch_size, n_steps, condition_horizon, n_conditions, d_model]
        
        attention_weights = q @ k.transpose(-1, -2) # [batch_size, n_steps, condition_horizon, m, n_conditions]
        attention_weights = F.softmax(input=(attention_weights / torch.sqrt(torch.tensor(d_model, dtype=torch.float32))), dim=-1) # [batch_size, n_steps, condition_horizon, m, n_conditions]: cross attention scores 
        attention_weights = self.dropout(attention_weights)
        
        out = attention_weights @ v # [batch, n_steps, condition_horizon, m, d_model] 
        out = self.norm(out)
        
        return out