import torch 
import torch.nn as nn 

class JointAttentionEncoder(nn.Module): 
    def __init__(self, description_input_dim, description_hidden_dim, description_out_dim, 
                 observation_in_dim, observation_out_dim, 
                 dynamic_joint_description, dynamic_joint_state, 
                 stability_epsilon, softmax_temperature_min, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.description_input_dim = description_input_dim
        self.description_hidden_dim = description_hidden_dim
        self.description_out_dim = description_out_dim
        self.observation_in_dim = observation_in_dim 
        self.observation_out_dim = observation_out_dim 

        self.dynamic_joint_description = dynamic_joint_description 
        self.dynamic_joint_state = dynamic_joint_state

        self.stability_epsilon = stability_epsilon
        self.joint_log_softmax_temperature = torch.log(self.softmax_temperature - self.softmax_temperature_min)
        self.softmax_temperature_min = softmax_temperature_min

        self.joint_descritption_encoder = nn.Sequential(
            nn.Linear(self.description_input_dim, self.description_hidden_dim), 
            nn.LayerNorm(), 
            nn.ELU(),
            nn.Linear(self.description_hidden_dim, self.description_out_dim)
            )
        
        self.joint_state_encoder = nn.Sequential(
            nn.Linear(self.observation_in_dim, self.observation_out_dim), 
            nn.ELU()
            )
        
    def forward(self): 
        d = self.joint_descritption_encoder(self.dynamic_joint_description)
        d = torch.clamp(nn.Tanh(d), -1.0 + self.stability_epsilon, 1.0 - self.stability_epsilon)

        latent_dynamic_joint_state = self.joint_state_encoder(self.dynamic_joint_state)

        joint_e_x = torch.exp(dynamic_joint_state_mask / (torch.exp(self.joint_log_softmax_temperature) + self.softmax_temperature_min))
        dynamic_joint_state_mask = joint_e_x / (joint_e_x.sum(axis=-1, keepdims=True) + self.stability_epsilon)
        dynamic_joint_state_mask = torch.repeat(torch.expand_dims(dynamic_joint_state_mask, axis=-1), latent_dynamic_joint_state.shape[-1], axis=-1)
        masked_dynamic_joint_state = dynamic_joint_state_mask * torch.expand_dims(latent_dynamic_joint_state, axis=-2)
        masked_dynamic_joint_state = torch.reshape(masked_dynamic_joint_state, masked_dynamic_joint_state.shape[:-2] + (masked_dynamic_joint_state.shape[-2] * masked_dynamic_joint_state.shape[-1],))
        dynamic_joint_latent = torch.sum(masked_dynamic_joint_state, axis=-2) # TODO: check if axis is correct 

