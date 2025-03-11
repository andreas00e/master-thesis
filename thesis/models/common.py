import torch
import torch.nn as nn

class Clip(nn.Module): 
    def __init__(self, stability_eps: float):
        super().__init__() 
        self.stability_eps = stability_eps
    
    def forward(self, x):
        return torch.clip(x, min=-1.0+self.stability_eps, max=1.0-self.stability_eps)

class ParametrizedSofmax(nn.Module):
    def __init__(self, softmax_temp: float, softmax_temp_min: float, joint_log_softmax_temp: float, stability_eps: float):
        super().__init__()
        self.softmax_temp = softmax_temp
        self.softmax_temp_min = softmax_temp_min
        self.joint_log_softmax_temp = joint_log_softmax_temp
        self.stability_eps = stability_eps
        
    def forward(self, x): 
        nom = torch.exp(x / (torch.exp(self.joint_log_softmax_temp) + self.softmax_temp_min))
        dom = torch.sum(nom, dim=-1, keepdim=True) + self.stability_eps
        return  nom / dom

class JointAttentionEncoder(nn.Module): 
    def __init__(self, desc_in_size: int, desc_out_size: int,
                 state_in_size: int, state_out_size: int, 
                 stability_eps: float, softmax_temp: float, softmax_temp_min: float, 
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.desc_in_size = desc_in_size
        self.desc_out_size = desc_out_size
        self.state_in_size = state_in_size 
        self.state_out_size = state_out_size 

        self.softmax_temp = softmax_temp
        self.softmax_temp_min = softmax_temp_min
        self.joint_log_softmax_temp = nn.Parameter(
            data=torch.log(torch.tensor(self.softmax_temp-self.softmax_temp_min)), requires_grad=True)
        self.stability_eps = stability_eps

        self.joint_desc_encoder = nn.Sequential(
            nn.Linear(self.desc_in_size, self.desc_out_size), 
            nn.LayerNorm(desc_out_size), 
            nn.ELU(),
            nn.Linear(self.desc_out_size, self.desc_out_size), 
            nn.Tanh(), 
            Clip(self.stability_eps), 
            ParametrizedSofmax(self.softmax_temp, self.softmax_temp_min, self.joint_log_softmax_temp, self.stability_eps)
            )
        
        self.joint_state_encoder = nn.Sequential(
            nn.Linear(self.state_in_size, self.state_out_size), 
            nn.ELU()
            )
    
    def forward(self, joint_desc, joint_state): 
        joint_desc_emb = self.joint_desc_encoder(joint_desc)
        joint_state_emb = self.joint_state_encoder(joint_state)

        dynamic_joint_state_mask = torch.repeat_interleave(
            torch.unsqueeze(joint_desc_emb, dim=-1), repeats=joint_state_emb.shape[-1], dim=-1)
        masked_dynamic_joint_state = dynamic_joint_state_mask * torch.unsqueeze(input=joint_state_emb, dim=-2)
        masked_dynamic_joint_state = torch.reshape(
            masked_dynamic_joint_state, shape=(masked_dynamic_joint_state.shape[:-2]+(masked_dynamic_joint_state.shape[-2] * masked_dynamic_joint_state.shape[-1],)))
        dynamic_joint_latent = torch.sum(masked_dynamic_joint_state, dim=-2)

        return dynamic_joint_latent