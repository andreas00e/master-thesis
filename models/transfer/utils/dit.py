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
        d_model: int,
        action_dim: int, 
        action_horizon: int, 
        obs_emb_dim: int, 
        noise_scheduler_kwargs: DictConfig, 
        decoder_layer_kwargs: DictConfig, 
        transformer_decoder_kwargs: DictConfig, 
        pe_kwargs: DictConfig
        ) -> None:
        super().__init__()
        
        self.d_model = d_model
        self.action_dim = action_dim 
        self.action_horizon = action_horizon
        self.obs_emb_dim = obs_emb_dim 

        self.noise_scheduler_kwargs = noise_scheduler_kwargs
        self.decoder_layer_kwargs = decoder_layer_kwargs 
        self.transformer_decoder_kwargs = transformer_decoder_kwargs
        self.pe_kwargs = pe_kwargs
        
        self.action_down = nn.Linear(self.action_dim, self.d_model)
        self.action_up = nn.Linear(self.d_model, self.action_dim)
        
        self.time_mlp = nn.Sequential( # TODO: Include Fourier embeddings
            nn.Linear(self.d_model, self.d_model), 
            nn.SiLU(), 
            nn.Linear(self.d_model, self.d_model)
        )
        
        self.scheduler = DDPMScheduler(**self.noise_scheduler_kwargs)
        self.pe = PE(**self.pe_kwargs)

        self.register_buffer("bos", torch.rand(1, 1, self.d_model)) # [1, 1, d_model]: Beginning of Sequence Token         
        self.register_buffer("eos", torch.rand(1, 1, self.d_model)) # [1, 1, d_model]: End of Sequence Token 

        self.decoder_layer = nn.TransformerDecoderLayer(**self.decoder_layer_kwargs)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, **self.transformer_decoder_kwargs)
        
        self.loss = nn.MSELoss()
        
    def forward(self, actions, rgb_emb, rce_emb):
        batch_size, n_steps, action_dim = actions.shape       
        
        timesteps = torch.randint(
            low=0, 
            high=self.noise_scheduler_kwargs.num_train_timesteps, 
            size=(actions.shape[0], ), 
            dtype=torch.int, 
            device=actions.device
            ) # [batch]
        
        noise = torch.randn_like(actions) # [batch, steps, action_dim]
        
        # Forward Process
        noisy_actions = self.scheduler.add_noise(original_samples=actions, noise=noise, timesteps=timesteps) # [batch, steps, action_dim]
         
        # Backward Process
        predicted_noise = self._backward_process(noisy_actions, rgb_emb, timesteps)
        predicted_noise = predicted_noise[:, 1:-1, :]
        loss = self.loss(noise, predicted_noise)
        
        return loss
    
    def _backward_process(
        self, 
        noisy_actions: TensorType["batch", "steps", "action_dim"], 
        condition: TensorType["batch", "steps", "*"], 
        timestep: TensorType["batch"], 
        ): 
        
        batch_size, n_steps, _ = noisy_actions.shape
                
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(
            sz=n_steps+2,  
            device=noisy_actions.device, 
            dtype=noisy_actions.dtype
            )
        
        noisy_actions_emb = self.action_down(noisy_actions) # [batch, steps, d_model]

        # Finding first padding value to insert <eos> at that point
        a_idxs = torch.isnan(noisy_actions_emb).all(dim=-1) # 
        a_idxs = torch.nonzero(a_idxs) # [n_steps, 2]
        
        first_pad_idxs = torch.full(size=(batch_size,), fill_value=n_steps+1, dtype=torch.long, device=noisy_actions_emb.device)        
        
        if a_idxs.numel() > 0: 
            first_pad_idxs = first_pad_idxs.scatter_reduce(dim=0, index=a_idxs[:, 0], src=a_idxs[:, 1], reduce="amin", include_self=True)
        
        self.bos = self.bos.expand(batch_size, -1, -1) # [batch, 1, d_model]
        self.eos = self.eos.expand(batch_size, -1, -1) # [batch, 1, d_model]
        
        pad = torch.full((batch_size, 1, self.d_model), fill_value=float("nan"), device=noisy_actions_emb.device) # [batch, 1, d_model]
        noisy_actions_emb = torch.concat(tensors=(self.bos, noisy_actions_emb, pad), dim=1) # [batch, 1+steps+1, d_model]
        condition = torch.concat(tensors=(self.bos, condition, pad), dim=1) # [batch, 1+steps+1, d_model]
        
        batch_idxs = torch.arange(0, batch_size, dtype=torch.int, device=noisy_actions_emb.device)              
        first_pad_idxs = first_pad_idxs.to(torch.int)
        
        noisy_actions_emb[batch_idxs, first_pad_idxs] = self.eos.squeeze() # [batch, d_model]
        condition[batch_idxs, first_pad_idxs] = self.eos.squeeze() # [batch, d_model]
        
        padding_mask = torch.isnan(noisy_actions_emb).all(dim=-1) # [batch, steps, d_model]
        noisy_actions_emb = self.pe(noisy_actions_emb) # TODO:: Include Positional Encoding !
        
        timestep = timestep.to(torch.float32)[:, None].repeat(1, self.d_model) # [batch, d_model]
        t_emb = self.time_mlp(timestep) # [batch, d_model]
        
        tgt = noisy_actions_emb + t_emb[:, None, :] # [batch, n_steps, d_model]

        out = self.decoder(
            tgt=tgt, 
            memory=condition,
            tgt_mask=tgt_mask,  
            tgt_key_padding_mask=padding_mask, 
            memory_key_padding_mask=padding_mask
            )
        
        predicted_noise = self.action_up(out)
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