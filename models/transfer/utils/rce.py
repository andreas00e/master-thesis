from typing import Dict, Any

import torch 
import torch.nn as nn 
from torchtyping import TensorType


class RCE(nn.Module):
    def __init__(
        self,
        dsc_kwargs: Dict[str, Any],
        obs_kwargs: Dict[str, Any],
        h_kwargs: Dict[str, Any],
    ) -> None:
        super().__init__()

        # retain original kwargs for flexibility
        self.dsc_kwargs = dsc_kwargs
        self.obs_kwargs = obs_kwargs
        self.h_kwargs = h_kwargs

        # learnable temperature offset (originally: tau - tau_min)
        init_tau = getattr(self.h_kwargs, "tau", None)
        init_tau_min = getattr(self.h_kwargs, "tau_min", 0.0)
        if init_tau is None:
            # fall back to dict-style access if attributes not present
            init_tau = self.h_kwargs.get("tau", 1.0)
            init_tau_min = self.h_kwargs.get("tau_min", 0.0)

        self.tau = nn.Parameter(torch.tensor(init_tau - init_tau_min), requires_grad=True)

        # encoders
        self.dsc_encoder = self._build_dsc_encoder()
        self.obs_encoder = self._build_obs_encoder()

    def _build_dsc_encoder(self) -> nn.Sequential:
        in_dim = getattr(self.dsc_kwargs, "input_dim", None) or self.dsc_kwargs.get("input_dim")
        hidden = getattr(self.dsc_kwargs, "hidden_dim", None) or self.dsc_kwargs.get("hidden_dim")
        out = getattr(self.dsc_kwargs, "output_dim", None) or self.dsc_kwargs.get("output_dim")

        return nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ELU(),
            nn.Linear(hidden, out),
        )

    def _build_obs_encoder(self) -> nn.Sequential:
        in_dim = getattr(self.obs_kwargs, "input_dim", None) or self.obs_kwargs.get("input_dim")
        out = getattr(self.obs_kwargs, "output_dim", None) or self.obs_kwargs.get("output_dim")

        return nn.Sequential(nn.Linear(in_dim, out), nn.ELU())

    def forward(self, joint_dsc: TensorType["*"], joint_obs: TensorType["*"]):
        # description embedding: [batch, n_joints, hidden]
        description_emb = self.dsc_encoder(joint_dsc)
        eps = getattr(self.h_kwargs, "epsilon", None)
        if eps is None:
            eps = self.h_kwargs.get("epsilon", 1e-6)

        description_emb = torch.clamp(torch.tanh(description_emb), -1.0 + eps, 1.0 - eps)

        # observation embedding and normalization
        tau_val = getattr(self.h_kwargs, "tau", None)
        if tau_val is None:
            tau_val = self.h_kwargs.get("tau", 1.0)

        observation_emb = self.obs_encoder(joint_obs)  # [batch, n_joints, emb]
        observation_emb = torch.exp(observation_emb / (tau_val + eps))
        observation_emb = observation_emb / torch.sum(observation_emb, dim=-1, keepdim=True)

        # combine and pool over joints
        emb = observation_emb * description_emb * observation_emb
        emb = torch.sum(emb, dim=1)
        return emb