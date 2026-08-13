# Diffusion Transformer (DIT)

from omegaconf import DictConfig

import torch 
import torch.nn as nn 
from torchtyping import TensorType

from diffusers import DDPMScheduler

from models.utils.utils import PE


class DIT(nn.Module): 
    def __init__(
        self, 
        action_dim: int, 
        obs_emb_dim: int, 
        action_horizon: int, 
        noise_scheduler_kwargs: DictConfig, 
        decoder_layer_kwargs: DictConfig, 
        transformer_decoder_kwargs: DictConfig, 
        pe_kwargs: DictConfig
        ) -> None:
        super().__init__()
        
        self.action_dim = action_dim 
        self.obs_emb_dim = obs_emb_dim 
        self.action_horizon = action_horizon
        
        self.noise_scheduler_kwargs = noise_scheduler_kwargs
        self.decoder_layer_kwargs = decoder_layer_kwargs 
        self.transformer_decoder_kwargs = transformer_decoder_kwargs
        self.pe_kwargs = pe_kwargs
        
        self.d_model = self.decoder_layer_kwargs.d_model
        self.action_in_proj = nn.Linear(self.action_dim, self.d_model)
        self.ation_out_proj = nn.Linear(self.d_model, self.action_dim)
        
        self.time_mlp = nn.Sequential(
            nn.Linear(self.d_model, self.d_model), 
            nn.SiLU(), 
            nn.Linear(self.d_model, self.d_model)
        )
        
        self.scheduler = DDPMScheduler(**self.noise_scheduler_kwargs)
        
        self.pe = PE(**self.pe_kwargs)

        self.decoder_layer = nn.TransformerDecoderLayer(**self.decoder_layer_kwargs)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, **self.transformer_decoder_kwargs)
        
        self.loss = nn.MSELoss()
        
    def forward(self, actions, condition): 
        timesteps = torch.randint(
            low=0, 
            high=self.noise_scheduler_kwargs.num_train_timesteps, 
            size=(actions.shape[0], ), 
            dtype=torch.int, 
            device=actions.device) # [batch]
        noise = torch.randn_like(actions) # [batch, steps, action_dim]
        
        # Forward Process
        noisy_actions = self.scheduler.add_noise(original_samples=actions, noise=noise, timesteps=timesteps) #  [batch, steps, action_dim]

        # Backward Process
        predicted_noise = self._backward_process(noisy_actions, condition, timesteps)
        
        loss = self.loss(predicted_noise)
        
        return loss
    
    
    def _backward_process(self, noisy_actions, condition, timestep): 
        noisy_actions_emb = self.action_in_proj(noisy_actions)
        noisy_actions_emb = self.pe(noisy_actions_emb)
        
        t_emb = self.time_mlp(timestep).unsqueeze(1)
        tgt = noisy_actions_emb + t_emb 
        
        memory = condition
        
        out = self.decoder(tgt=tgt, memory=memory)
        
        predicted_noise = self.action_out_proj(out)
        return predicted_noise
    
    
    # @torch.no_grad()
    # def sample_action(self, obs, num_samples=1):
    #     batch_size = obs.shape[0]
    #     device = obs.device

    #     # Step 1: start from pure Gaussian noise
    #     action_flat = torch.randn(
    #         batch_size, num_samples,
    #         self.action_horizon * self.action_dim, device=device
    #     )

    #     # Step 2: count down from T=100 to 0, removing a little noise each step
    #     for t in reversed(range(self.num_diffusion_steps)):
    #         timestep = torch.full((batch_size,), t, device=device, dtype=torch.long)

    #         # Ask the network: "what noise is present at this timestep?"
    #         predicted_noise = torch.stack([
    #             self.forward(obs, action_flat[:, i], timestep)
    #             for i in range(num_samples)
    #         ], dim=1)

    #         # Remove that noise (DDPM update rule)
    #         alpha_t      = self.alphas_cumprod[t]
    #         alpha_t_prev = self.alphas_cumprod[t-1] if t > 0 else torch.tensor(1.0)
    #         action_flat  = (action_flat - (1 - alpha_t).sqrt() * predicted_noise) / alpha_t.sqrt()

    #         # Add a small amount of fresh noise (except at the very last step)
    #         if t > 0:
    #             action_flat = (alpha_t_prev.sqrt() * action_flat
    #                            + (1 - alpha_t_prev).sqrt() * torch.randn_like(action_flat))

    #     # Step 3: reshape → (batch, num_samples, action_horizon, action_dim)
    #     return action_flat.reshape(batch_size, num_samples, self.action_horizon, self.action_dim)