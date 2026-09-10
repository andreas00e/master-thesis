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
        
        # Action projection 
        self.action_down = nn.Linear(self.action_dim, self.d_model)
        self.action_up = nn.Linear(self.d_model, self.action_dim)
        
        self.time_mlp = nn.Sequential( # TODO: Include Fourier embeddings
            nn.Linear(self.d_model, self.d_model), 
            nn.SiLU(), 
            nn.Linear(self.d_model, self.d_model)
        )
        
        self.scheduler = DDPMScheduler(**self.noise_scheduler_kwargs)
        self.pe = PE(**self.pe_kwargs)
        
        self.bos = nn.Parameter(torch.empty(1, 1, self.d_model)) # Beginning of Sequence Token      
        self.eos = nn.Parameter(torch.empty(1, 1, self.d_model)) # End of Sequence Token
        nn.init.xavier_uniform_(self.bos)
        nn.init.xavier_uniform_(self.eos)

        self.decoder_layer = nn.TransformerDecoderLayer(**self.decoder_layer_kwargs)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, **self.transformer_decoder_kwargs)
        
        self.loss = nn.MSELoss() # XXX: Do I really need this? 
        self.softmax = nn.Softmax()
        
    def forward(self, actions: TensorType["batch", "steps", "d_model"], conditions: TensorType["batch", "steps", "k", "d_model"]) -> TensorType["1"]:        
        batch_size, n_steps, _ = actions.shape
        timesteps = torch.randint(
            low=0, 
            high=self.noise_scheduler_kwargs.num_train_timesteps, 
            size=(batch_size, ),
            dtype=torch.long, 
            device=actions.device
            ) # [batch]
        
        noise = torch.randn_like(actions) # [batch, steps, action_dim]
        
        # Forward process: Add noise to input sample 
        noisy_actions = self._forward_process(actions, noise, timesteps) # [batch, steps, action_dim]
         
        # Backward process
        predicted_noise = self._backward_process(noisy_actions, conditions, timesteps)
        predicted_noise = predicted_noise[:, 1:-1, :]
        
        loss = self.loss(noise, predicted_noise)
        
        return loss
    
    def _forward_process(
        self, 
        actions: TensorType["batch", "steps", "action_dim"],
        noise: TensorType["batch", "steps", "action_dim"], 
        timesteps: TensorType["batch"]
        ) -> TensorType["batch", "steps", "actions_dim"]: 

        noisy_sample = self.scheduler.add_noise(original_samples=actions, noise=noise, timesteps=timesteps) # [batch, steps, d_model] 
        
        return noisy_sample 
    
    def _backward_process(
        self, 
        noisy_actions: TensorType["batch", "steps", "action_dim"], 
        conditions: TensorType["batch", "steps", "k", "d_model"], 
        timesteps: TensorType["batch"], 
        ) -> TensorType["batch", "n_steps", "action_dim"]: 
        
        batch_size, n_steps, _ = noisy_actions.shape
        conditions = torch.sum(conditions, dim=-2) # [batch, steps, d_model]
                
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(
            sz=n_steps+2,  
            device=noisy_actions.device, 
            dtype=noisy_actions.dtype
            ) # [steps, steps]
        
        noisy_actions_emb = self.action_down(noisy_actions) # [batch, steps, d_model]

        # Finding first padding value to insert <eos> at that point
        a_idxs = torch.isnan(noisy_actions_emb).all(dim=-1) # [batch, steps]
        a_idxs = torch.nonzero(a_idxs) # [batch, steps]
        
        first_pad_idxs = torch.full(size=(batch_size,), fill_value=n_steps+1, dtype=torch.long, device=noisy_actions_emb.device)        
        
        if a_idxs.numel() > 0: 
            first_pad_idxs = first_pad_idxs.scatter_reduce(dim=0, index=a_idxs[:, 0], src=a_idxs[:, 1], reduce="amin", include_self=True)
        
        bos = self.bos.clone().expand(batch_size, -1, -1) # [batch, 1, d_model]
        eos = self.eos.clone().expand(batch_size, -1, -1) # [batch, 1, d_model]
        
        pad = torch.full((batch_size, 1, self.d_model), fill_value=float("nan"), device=noisy_actions_emb.device) # [batch, 1, d_model]
        noisy_actions_emb = torch.concat(tensors=(bos, noisy_actions_emb, pad), dim=1) # [batch, 1+steps+1, d_model]
        conditions = torch.concat(tensors=(bos, conditions, pad), dim=1) # [batch, 1+steps+1, d_model]
        
        batch_idxs = torch.arange(0, batch_size, dtype=torch.int, device=noisy_actions_emb.device)              
        first_pad_idxs = first_pad_idxs.to(torch.int)
        
        noisy_actions_emb[batch_idxs, first_pad_idxs] = eos.squeeze()
        conditions[batch_idxs, first_pad_idxs] = eos.squeeze()
        
        padding_mask = torch.isnan(noisy_actions_emb).all(dim=-1) # [batch, 1+steps+1, d_model]
        noisy_actions_emb = self.pe(noisy_actions_emb) # [batch, 1+steps+1, d_model]
        conditions = self.pe(conditions) # [batch, 1+steps+1, d_model]
        
        timesteps = timesteps.to(torch.float32)[:, None].repeat(1, self.d_model) # [batch, d_model]
        t_emb = self.time_mlp(timesteps) # [batch, d_model]
        tgt = noisy_actions_emb + t_emb[:, None, :] # [batch, 1+steps+1, d_model]

        out = self.decoder(
            tgt=tgt, 
            memory=conditions,
            tgt_mask=tgt_mask,  
            tgt_key_padding_mask=padding_mask, 
            memory_key_padding_mask=padding_mask
            )
        
        predicted_noise = self.action_up(out) # [batch, n_steps, action_dim]
        return predicted_noise
    
    @torch.no_grad()
    def sample(self, rgb_emb):
        batch_size = rgb_emb.shape[0]

        # Start from pure Gaussian noise 
        noisy_actions = torch.randn(size=(batch_size, self.action_horizon, self.action_dim), dtype=rgb_emb.dtype, device=rgb_emb.device) # [batch_size, action_horizon, action_dim]
        
        # Iteratively remove noise from sample 
        for t in reversed(range(self.noise_scheduler_kwargs.num_train_timesteps)): #  [99, 98, ..., 0]
            timesteps = torch.full(size=(batch_size, ), fill_value=t, dtype=torch.long, device=rgb_emb.device) # [batch]
            # Predicted noise based on current sample 
            predicted_noise = self._backward_process(noisy_actions=noisy_actions, conditions=rgb_emb, timesteps=timesteps)
            # Scheduler output
            step = self.scheduler.step(model_output=predicted_noise, timestep=timesteps, sample=noisy_actions) 
            # Reconstruct previous sample in diffusion process
            noisy_actions = step.prev_sample
    
        return noisy_actions