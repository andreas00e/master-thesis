from typing import Optional
from omegaconf import DictConfig

import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torchtyping import TensorType

from diffusers import DDPMScheduler
from models.utils.utils import PositionalEncoding, SinusoidalEmbedding


class DiffusionTransformer(nn.Module): 
    def __init__(
        self, 
        d_model: int,
        action_dim: int, 
        action_horizon: int, 
        condition_horizon: int, 
        noise_scheduler_kwargs: DictConfig, 
        decoder_layer_kwargs: DictConfig, 
        transformer_decoder_kwargs: DictConfig, 
        pe_kwargs: DictConfig
        ) -> None:
        super().__init__()
        
        self.d_model = d_model
        self.action_dim = action_dim 
        self.action_horizon = action_horizon
        self.condition_horizon = condition_horizon
        
        self.noise_scheduler_kwargs = noise_scheduler_kwargs
        self.decoder_layer_kwargs = decoder_layer_kwargs 
        self.transformer_decoder_kwargs = transformer_decoder_kwargs
        self.pe_kwargs = pe_kwargs
        
        # Action projection 
        self.action_down = nn.Linear(self.action_dim, self.d_model)
        self.action_up = nn.Linear(self.d_model, self.action_dim)
        
        self.time_emb = nn.Sequential(
            nn.Linear(self.d_model, self.d_model), 
            nn.SiLU(), 
            nn.Linear(self.d_model, self.d_model)
        )
        
        self.scheduler = DDPMScheduler(**self.noise_scheduler_kwargs)
        self.positional_encoding = PositionalEncoding(**self.pe_kwargs)
        self.sinusoidal_embedding = SinusoidalEmbedding(d_new=self.d_model)
        
        self.decoder_layer = nn.TransformerDecoderLayer(**self.decoder_layer_kwargs)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, **self.transformer_decoder_kwargs)
        
    def forward(
        self, 
        actions: TensorType["batch", "action_horizon", "action_dim"], # self.action_horizon actions to predict 
        conditions: TensorType["batch", "obs_horizon", "k", "d_model"], # last self.condition_horizon observations
        actions_idxs: TensorType["batch", "action_horizon"], 
        conditions_idxs: TensorType["batch", "obs_horizon"]
        ) -> torch.Tensor:   
             
        batch_size = actions.shape[0]
        
        timesteps = torch.randint(
            low=0, 
            high=self.noise_scheduler_kwargs.num_train_timesteps, 
            size=(batch_size, ),
            dtype=torch.long, 
            device=actions.device
            ) # [batch]
        
        noise = torch.randn_like(actions) # [batch, action_horizon, action_dim]
        padding_mask = torch.all(torch.isnan(actions), dim=-1) # [batch, action_horizon]
        
        # Forward process: Add noise to input sample 
        noisy_actions = self._forward_process(actions, noise, timesteps) # [batch, n_steps, action_dim]
         
        # Backward process
        predicted_noise = self._backward_process(noisy_actions, conditions, actions_idxs, conditions_idxs, timesteps, padding_mask)
        
        valid_mask = (~padding_mask).float()
        loss_nom = torch.sum(torch.mean((noise - predicted_noise) ** 2, dim=-1) * valid_mask, dim=-1) # [batch]
        loss_denom = torch.sum(valid_mask, dim=-1) # [batch]
        loss = torch.mean(loss_nom / loss_denom)
        
        return loss
    
    def _forward_process(
        self, 
        actions: TensorType["batch", "action_horizon", "action_dim"],
        noise: TensorType["batch", "action_horizon", "action_dim"], 
        timesteps: TensorType["batch"]
        ) -> TensorType["batch", "action_horizon", "action_dim"]: 

        noisy_sample = self.scheduler.add_noise(original_samples=actions, noise=noise, timesteps=timesteps) # [batch, n_steps, action_dim] 
        
        return noisy_sample 
    
    def _backward_process( # reconstruct sample from random noise
        self, 
        noisy_actions: TensorType["batch", "action_horizon", "action_dim"], 
        conditions: TensorType["batch", "condition_horizon", "k", "d_model"], # k: number of condition modalities
        actions_idxs: TensorType["batch", "action_horizon"], 
        conditions_idxs: TensorType["batch", "condition_horizon"], 
        timesteps: TensorType["batch"], 
        padding_mask: Optional[TensorType["batch", "action_horizon"]]=None, 
        ) -> TensorType["batch", "action_horizon", "action_dim"]: 
                        
        conditions = torch.sum(conditions, dim=-2) # [batch, condition_horizon, d_model]
        conditions = self.positional_encoding(conditions, idxs=conditions_idxs)
        
        noisy_actions_emb = self.action_down(noisy_actions) # [batch, action_horizon, d_model]
        noisy_actions_emb = self.positional_encoding(noisy_actions_emb, actions_idxs) # [batch, n_steps, d_model]
        
        timesteps = timesteps.unsqueeze(-1).to(torch.float32) # [batch, 1]
        timesteps = self.sinusoidal_embedding(timesteps) # [batch, d_model]
        timesteps_emb = self.time_emb(timesteps).unsqueeze(1) # [batch, 1, d_model]
        tgt = noisy_actions_emb + timesteps_emb # [batch, n_steps, d_model] 
        
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(self.action_horizon)

        out = self.decoder(
            tgt=tgt, # self.action_horizon to predict actions 
            memory=conditions, # self.condition_horizon last observations
            tgt_mask=tgt_mask, 
            tgt_key_padding_mask=padding_mask, 
            memory_key_padding_mask=padding_mask
            ) # [batch, n_steps, d_model]
        
        predicted_noise = self.action_up(out) # [batch, n_steps, action_dim]
        
        return predicted_noise
    
    @torch.no_grad()
    def sample(
        self, 
        conditions: TensorType["batch", "condition_horizon", "k", "d_model"], 
        actions_idxs: TensorType["batch", "action_horizon"], 
        conditions_idxs: TensorType["batch", "condition_horizon"]
        ) -> TensorType["batch", "action_horizon", "action_dim"]:
        self.eval() 
        
        batch_size = conditions.shape[0]

        # Start from pure Gaussian noise 
        noisy_actions = torch.randn(size=(batch_size, self.action_horizon, self.action_dim), dtype=conditions.dtype, device=conditions.device) # [batch_size, action_horizon, action_dim]
        
        # Iteratively remove noise from sample 
        for t in reversed(range(self.noise_scheduler_kwargs.num_train_timesteps)): #  [99, 98, ..., 0]
            timesteps = torch.full((batch_size, ), fill_value=t, dtype=torch.long, device=conditions.device) # []
            # Predicted noise based on current sample 
            
            predicted_noise = self._backward_process(
                noisy_actions=noisy_actions,
                conditions=conditions, 
                actions_idxs=actions_idxs, 
                conditions_idxs=conditions_idxs, 
                timesteps=timesteps
                )
            
            # Scheduler output
            step = self.scheduler.step(model_output=predicted_noise, timestep=t, sample=noisy_actions) 
            # Reconstruct previous sample in diffusion process
            noisy_actions = step.prev_sample
         
        self.train()   
        return noisy_actions