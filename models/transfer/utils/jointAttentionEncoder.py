from typing import Dict

import torch 
import torch.nn as nn 
from torchtyping import TensorType


class JointAttentionEncoder(nn.Module): 
    def __init__(self, 
        description_kwargs: Dict,  
        observation_kwargs: Dict, 
        entropy_kwargs: Dict,
        *args, **kwargs) -> None: 
        super().__init__()
        
        self.description_kwargs = description_kwargs
        self.observation_kwargs = observation_kwargs
        self.entropy_kwargs = entropy_kwargs
 
        self.tau = nn.Parameter(
            data=torch.tensor(
                self.entropy_kwargs.tau-self.entropy_kwargs.tau_min
                ), requires_grad=True
            )

        self.description_encoder = nn.Sequential(
            nn.Linear(
                self.description_kwargs.input_dim, 
                self.description_kwargs.hidden_dim
                ), 
            nn.LayerNorm(
                self.description_kwargs.hidden_dim
                ), 
            nn.ELU(),
            nn.Linear(
                self.description_kwargs.hidden_dim, 
                self.description_kwargs.output_dim
                )
            )
        
        self.observation_encoder = nn.Sequential(
            nn.Linear(
                self.observation_kwargs.input_dim, 
                self.observation_kwargs.output_dim
                ), 
            nn.ELU()
            )

    def forward(self, joint_descriptions: TensorType["*"], joint_observations: TensorType["*"]) -> TensorType["batch", "1"]:
        description_emb = self.description_encoder(joint_descriptions)
        description_emb = torch.clamp(
            nn.Tanh(description_emb), -1.0 + self.entropy_kwargs.epsilon, 1.0 + self.entropy_kwargs.epsilon
            )

        observation_emb = self.observation_encoder(joint_observations) # [batch, n_joints, dim] -> [batch, n_joints, emb]
        observation_emb = torch.exp(observation_emb / (self.entropy_kwargs.tau + self.entropy_kwargs.epsilon)) # [batch, n_joints, emb]
        observation_emb = observation_emb / torch.sum(observation_emb, dim=-1) # [batch, n_joints, hidden_dim]
        
        emb = observation_emb * description_emb * observation_emb  # [batch, n_joints, hidden_dim]
        emb = torch.sum(emb, dim=1)  # [batch, hidden_dim]
        
        return emb 
    
    def training_step(self, batch: TensorType["*"], batch_idx: TensorType["*"]) -> TensorType["*"]:
        joint_description, joint_observation = batch.values()
        emb = self.forward(joint_description, joint_observation)
        return emb 