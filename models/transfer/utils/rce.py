# Robot Conditioned Encoder (RCE)

from omegaconf import DictConfig 

import torch 
import torch.nn as nn 
from torchtyping import TensorType


class RCE(nn.Module):
    def __init__(
        self,
        dsc_kwargs: DictConfig,
        obs_kwargs: DictConfig,
        h_kwargs: DictConfig
        ) -> None:
        super().__init__()

        self.dsc_kwargs = dsc_kwargs
        self.obs_kwargs = obs_kwargs
        self.h_kwargs = h_kwargs
        
        self.dsc_encoder = self._build_dsc_encoder() # joint description encoder 
        self.obs_encoder = self._build_obs_encoder() # joint observation encoder

        self.init_tau = getattr(self.h_kwargs, "tau", 1.0)
        self.init_tau_min = getattr(self.h_kwargs, "tau_min", 0.0)
        self.epsilon = getattr(self.h_kwargs, "epsilon", 1e-6)

        self.tau = nn.Parameter(torch.tensor(self.init_tau - self.init_tau_min), requires_grad=True)

    def _build_dsc_encoder(self) -> nn.Sequential:
        input_dim = getattr(self.dsc_kwargs, "input_dim", None) 
        hidden_dim = getattr(self.dsc_kwargs, "hidden_dim", None)
        output_dim = getattr(self.dsc_kwargs, "output_dim", None)
        
        if any([isinstance(dim, type(None)) for dim in [input_dim, hidden_dim, output_dim]]): 
            raise ValueError("Linear layer dimensions of the description encoder cannot be zero!")
        else:   
            return nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ELU(),
                nn.Linear(hidden_dim, output_dim)
                )

    def _build_obs_encoder(self) -> nn.Sequential:
        input_dim = getattr(self.obs_kwargs, "input_dim", None) 
        output_dim = getattr(self.obs_kwargs, "output_dim", None)
        
        if any([isinstance(dim, type(None)) for dim in [input_dim, output_dim]]): 
            raise ValueError("Linear layer dimensions of the observation encoder cannot be zero!")
        else:  
            return nn.Sequential(
                nn.Linear(input_dim, output_dim), 
                nn.ELU()
                )

    def forward(self, joint_dsc: TensorType["*"], joint_obs: TensorType["*"]):
        dsc_emb = self.dsc_encoder(joint_dsc)
        dsc_emb = torch.clamp(torch.tanh(dsc_emb), -1.0 + self.eps, 1.0 - self.eps)

        obs_emb = self.obs_encoder(joint_obs)  # [batch, n_joints, emb]
        obs_emb = torch.exp(obs_emb / (self.tau.data + self.epsilon))
        obs_emb = obs_emb / torch.sum(obs_emb, dim=-1, keepdim=True)

        emb = obs_emb * dsc_emb * obs_emb
        emb = torch.sum(emb, dim=1)
        return emb