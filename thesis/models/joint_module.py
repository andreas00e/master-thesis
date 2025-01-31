import torch 
import torch.nn as nn 

class JointAttentionEncoder(nn.Module): 
    def __init__(self, descr_in_size, descr_hidden_size, descr_out_size, 
                 obs_in_size, obs_out_size, 
                 stability_epsilon, softmax_temperature, softmax_temperature_min, 
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.descr_in_dim = descr_in_size
        self.descr_hid_dim = descr_hidden_size
        self.descr_out_dim = descr_out_size
        self.obs_in_dim = obs_in_size 
        self.obs_out_dim = obs_out_size 
    
        self.softmax_temperature = softmax_temperature
        self.softmax_temperature_min = softmax_temperature_min
        self.joint_log_softmax_temperature = nn.Parameter(data=torch.tensor(self.softmax_temperature-self.softmax_temperature_min), requires_grad=True)
        self.stability_epsilon = stability_epsilon

        self.joint_descritption_encoder = nn.Sequential(
            nn.Linear(self.descr_in_dim, self.descr_hid_dim), 
            nn.LayerNorm(descr_hidden_size), 
            nn.ELU(),
            nn.Linear(self.descr_hid_dim, self.descr_out_dim)
            )
        
        self.joint_state_encoder = nn.Sequential(
            nn.Linear(self.obs_in_dim, self.obs_out_dim), 
            nn.ELU()
            )
        
    def forward(self, dynamic_joint_descr, dynamic_joint_state): 
        dynamic_joint_latent = self.joint_descritption_encoder(dynamic_joint_descr)
        dynamic_joint_latent = torch.clamp(nn.Tanh(dynamic_joint_latent), -1.0 + self.stability_epsilon, 1.0 - self.stability_epsilon)

        latent_dynamic_joint_state = self.joint_state_encoder(dynamic_joint_state)

        joint_e_x = torch.exp(dynamic_joint_latent / (torch.exp(self.joint_log_softmax_temperature) + self.softmax_temperature_min))
        dynamic_joint_latent = joint_e_x / (joint_e_x.sum(axis=-1, keepdims=True) + self.stability_epsilon)
        dynamic_joint_state_mask = torch.repeat_interleave(torch.unsqueeze(dynamic_joint_state_mask, dim=-1), latent_dynamic_joint_state.shape[-1], axis=-1)
        masked_dynamic_joint_state = dynamic_joint_state_mask * torch.unsqueeze(latent_dynamic_joint_state, axis=-2)
        masked_dynamic_joint_state = torch.reshape(masked_dynamic_joint_state, masked_dynamic_joint_state.shape[:-2] + (masked_dynamic_joint_state.shape[-2] * masked_dynamic_joint_state.shape[-1],))
        dynamic_joint_latent = torch.sum(masked_dynamic_joint_state, axis=-2)

        return dynamic_joint_latent

