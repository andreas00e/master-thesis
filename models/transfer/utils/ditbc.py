from omegaconf import DictConfig

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtyping import TensorType

from diffusers import DDPMScheduler

from models.utils.utils import PE


class Transformer(nn.Module): 
    def __init__(
        self, 

        te_layer_kwargs: DictConfig,
        te_kwargs: DictConfig, 
        pe_kwargs: DictConfig
        ):
        super().__init__()
        
        self.te_layer_kwargs = te_layer_kwargs
        self.te_kwargs = te_kwargs
        self.pe_kwargs = pe_kwargs
        
        self.emb = nn.Linear(in_features=512, out_features=512)
        
        self.te_layer = nn.TransformerEncoderLayer(**self.te_layer_kwargs)
        self.te = nn.TransformerEncoder(self.te_layer, **self.te_kwargs)
        
        self.pe = PE(**self.pe_kwargs)
    
    def forward(self, x): 
        x = self.emb(x)
        x = torch.vstack(x, self.pe)
        x = self.te(x)
        return x 
    

class DiTBC(nn.Module):
    def __init__(
        self, 
        action_dim: int, 
        obs_dim: int, 
        horizon: int, 
        noise_scheduler_kwargs,
        hidden_dim: int,
        num_heads: int, 
        num_layers: int,
        num_timesteps: int,
        device: str, 
        
        ) -> None:
        super().__init__()
        
        self.action_dim = action_dim
        self.obs_dim = obs_dim 
        self.horizon = horizon
        self.noise_scheduler_kwargs = noise_scheduler_kwargs
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads 
        self.num_layers = num_layers 
        self.num_timesteps = num_timesteps
        
        self.device = device

        # Input projections (Actions to Tokens)
        self.action_embeder = nn.Linear(action_dim, hidden_dim)
        self.pos_embedding = nn.Parameter(torch.zeros(1, horizon, hidden_dim))

        # Conditioning context networks
        self.time_mlp = nn.Sequential(
            self.pe(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.obs_projector = nn.Linear(obs_dim, hidden_dim)
        self.cond_fusion = nn.Linear(hidden_dim, hidden_dim)

        # DiT Transformer Backbone layers
        self.blocks = Transformer()

        # Final projection mapping tokens back to raw action parameters
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.action_head = nn.Linear(hidden_dim, action_dim)

        
        self.noise_scheduler = DDPMScheduler(**self.noise_scheduler_kwargs)
        # DDPM Constants (Linear Variance Scheduler)
        beta = torch.linspace(self.beta[0], self.beta[1], num_timesteps)
        alpha = 1.0 - beta
        alpha_hat = torch.cumprod(alpha, dim=0)
        self.register_buffer("beta", beta)
        self.register_buffer("alpha", alpha)
        self.register_buffer("alpha_hat", alpha_hat)

    def _forward_dit(self, x_t: torch.Tensor, t: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        """Helper to run the Transformer sequence given noisy inputs and conditions."""
        B, T, _ = x_t.shape
        
        # 1. Tokenize action sequence and inject learnable position embeddings
        tokens = self.action_embeder(x_t) + self.pos_embedding[:, :T, :]

        # 2. Synthesize conditional metadata (Time + Observation State context)
        t_emb = self.time_mlp(t)   # [B, hidden_dim]
        obs_emb = self.obs_projector(obs)  # [B, hidden_dim]
        global_cond = self.cond_fusion(t_emb + obs_emb).unsqueeze(1) # [B, 1, hidden_dim]

        # 3. Process tokens dynamically via adaptive layer norm transformer blocks
        for block in self.blocks:
            tokens = block(tokens, global_cond)

        # 4. Map output tokens back into action dimensional changes
        return self.action_head(self.final_norm(tokens))

    def _loss(self, actions: TensorType["b", "n", "d"], obs: TensorType["b", "n", "d"]) -> TensorType["*"]: 
        b, n, _ = actions.shape # [b, n, ]
        t = torch.randint(0, self.num_timesteps, (n,), device=self.device)
        noise = torch.randn_like(actions)
        noisy_x = self.noise_scheduler.add_noise(actions, noise, t)

        # Apply noise mapping using standard forward schedule formula
        alpha_hat_t = self.alpha_hat[t].view(b, 1, 1)
        x_t = torch.sqrt(alpha_hat_t) * actions + torch.sqrt(1.0 - alpha_hat_t) * noise

        # Calculate error delta on structural variance
        predicted_noise = self._forward_dit(x_t, t, obs)
        return F.mse_loss(predicted_noise, noise)

    @torch.no_grad()
    def sample(self, obs: torch.Tensor) -> torch.Tensor:
        """Executes backwards denoising pass generating complete trajectories from noise."""
        B = obs.shape[0]
        device = obs.device

        # Sample initial raw normal Gaussian sequence blocks
        x_t = torch.randn((B, self.horizon, self.action_dim), device=device)

        # Backwards recursion sequence
        for t in reversed(range(self.num_timesteps)):
            t_batch = torch.full((B,), t, device=device, dtype=torch.long)
            pred_noise = self._forward_dit(x_t, t_batch, obs)

            alpha_t = self.alpha[t]
            alpha_hat_t = self.alpha_hat[t]
            beta_t = self.beta[t]

            # Reconstruct the mean denoised step
            mean = (1 / torch.sqrt(alpha_t)) * (x_t - (beta_t / torch.sqrt(1.0 - alpha_hat_t)) * pred_noise)
            
            if t > 0:
                noise = torch.randn_like(x_t)
                x_t = mean + torch.sqrt(beta_t) * noise
            else:
                x_t = mean

        return x_t

if __name__ == "__main__":
    BATCH_SIZE = 16
    OBS_DIM = 32         # Dimension of observation representation features
    ACTION_DIM = 7       # e.g., 7-DoF Robot joint configuration targets
    TRAJECTORY_LEN = 16  # Dynamic planning action output sequence chunk
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Instantiate model
    dit_policy = DiTBC(
        action_dim=ACTION_DIM, 
        obs_dim=OBS_DIM, 
        horizon=TRAJECTORY_LEN
    ).to(device)
    
    optimizer = torch.optim.AdamW(dit_policy.parameters(), lr=2e-4, weight_decay=1e-4)

    # Mock Expert Data (Actions normalized between [-1, 1])
    expert_actions = torch.tanh(torch.randn(BATCH_SIZE, TRAJECTORY_LEN, ACTION_DIM, device=device))
    env_observations = torch.randn(BATCH_SIZE, OBS_DIM, device=device)

    # Train Step Mock
    dit_policy.train()
    optimizer.zero_grad()
    loss = dit_policy._loss(expert_actions, env_observations)
    loss.backward()
    optimizer.step()
    print(f"DiT Step Training complete. Loss value: {loss.item():.4f}")

    # Inference Execution Block
    dit_policy.eval()
    test_obs = torch.randn(1, OBS_DIM, device=device)
    generated_trajectory = dit_policy.sample(test_obs)
    print(f"Generated Action Trajectory Shape: {generated_trajectory.shape}") # Expect [1, 16, 7]
