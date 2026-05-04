import torch 
import torch.nn as nn
import torch.nn.init as init

import r3m 

import wandb 
from termcolor import colored


class FFCombiner(nn.Module): 
    def __init__(self, hidden_size, out_size):
        super().__init__()
        self.obs_down = nn.Linear(hidden_size, out_size)
    
    def forward(self, input): 
        return nn.ReLU(self.obs_down(input))
    
class TemperatureCombiner(nn.Module):
    def __init__(self, batch_size: int, hidden_size:int):
        super().__init__()       
        # Learnable temperature  
        self.T = nn.Parameter(data=torch.ones(size=(batch_size, hidden_size)), requires_grad=True) # (B, hidden_size)
    
    def forward(self, emb1, emb2):
        assert emb1.shape == emb2.shape, colored("Dimensions of given tensors do not match", 'red')
        d = torch.tensor(emb1.shape[-1]) # Dimenison of input for normalization
        z = emb1 * emb2 / torch.sqrt(d) # (B, hidden_size)
        logit = torch.exp(z / self.T) # (B, hidden_size)
        out = logit / torch.sum(logit, dim=-1, keepdim=True) # (B, hidden_size)
        return out, self.T
    
class VisionEncoder(nn.Module): 
    def __init__(self, hidden_size: int, version: str, logger, device: str='cuda', combine: str='temperature'):
        super().__init__()
        self.hidden_size = hidden_size
        self.logger = logger
        self.device = device
        self.combine = combine
        self.img_emb = r3m.load_r3m(version)
        self.img_emb.eval().to(self.device)
    
    def forward(self, input): # input: (B, C, H, W)
        assert input.shape[1]%3 == 0, 'Given number of channels is no multiple of 3!'
                
        if input.shape[1] == 3: 
            obs_emb = self.img_emb(input) # (B, out_size)
        else: # more than one camera view/ more than three channels 
            n_images = int(input.shape[1]/3)
            
            obs_embs = []
            for i in range(n_images): 
                input_slice = input[:, i * 3 : (i + 1) * 3, :, :]
                obs_shape = input_slice.shape[1:] # (C, W, H)
                obs_emb = self.img_emb(input_slice, obs_shape=list(obs_shape)) # shape of input needs to be passed to r3m for it to know how to preprocess the given input 
                print(obs_emb.shape)
                obs_embs.append(obs_emb) # [(B, out_size), (B, out_size), ... (B, out_size)]
            
            if self.combine == 'ff':
                obs_emb = torch.cat(obs_embs, dim=1) # (B, out_size*n_images)
                ffcombiner = FFCombiner(hidden_size=obs_emb.shape[1], out_size=self.hidden_size)
            elif self.combine == 'temperature': 
                # TODO: Implement logic supporting more than two views
                temperatureCombiner = TemperatureCombiner(batch_size=input.shape[0], hidden_size=obs_embs[0].shape[-1]).to(self.device) 
                print(colored(f"Combining images from different views using softmax with temperature.", 'green'))
                obs_emb, T = temperatureCombiner(emb1=obs_embs[0], emb2=obs_embs[1]) # obs_emb: (B, hidden_size)
                self.logger.log(
                    {'temperature': wandb.Histogramm(T.detach().cpu().numpy())}
                )
                ffcombiner = FFCombiner(hidden_size=obs_emb.shape[1], out_size=self.hidden_size).to(self.device)

        return ffcombiner(obs_emb)
        

class VisionCombiner(nn.Module): 
    def __init__(self, hidden_size, out_size, resnet_version: str='resnet18', device: str='cuda'):
        super().__init__()
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.resnet_version = resnet_version
        self.device = device 
        
        self.img_emb = r3m.load_r3m(self.resnet_version)
        
        self.rgb_down = nn.Linear(self.hidden_size, self.out_size)
        self.depth_down = nn.Linear(self.hidden_size, self.out_size)
    
    def forward(self, input):
        view_embs = []
        for key in input.keys(): # input: (B, horizon, C=4 or C=6, H, W)
            batch_size, image_horizon = input[key].shape[0], input[key].shape[1]
            
            rgb_images = input[key][:, :, :3, ...].view(batch_size*image_horizon, 3, *input[key].shape[3:]) # (B*horizon, C=3, H, W)
            rgb_emb = self.img_emb(rgb_images) # (B*horizon, hidden_size)
            rgb_emb = self.rgb_down(rgb_emb) # (B*horizon, out_size)
            rgb_emb = rgb_emb.view(batch_size, image_horizon, self.out_size) # (B, horizon, out_size)
        
            if input[key].shape[2] == 4: # One channel -> Depth map needs to be expanded along channel dimension
                depth_images = input[key][:, :, 3, ...].unsqueeze(2).expand(-1, -1, 3, -1, -1) # (B, horizon, C=3, H, W)
            elif input[key].shape[2] == 6: 
                depth_images = input[key][:, :, 3:, ...] # (B, horizon, C=3, H, W)
            else: 
                raise ValueError("Depth map with unknown channel dimension!")
            
            depth_images = depth_images.view(batch_size*image_horizon, 3, *input[key].shape[3:]) # (B*horizon, C=3, H, W)
            depth_emb = self.img_emb(depth_images) # (B*horizon, hidden_size)
            depth_emb = self.rgb_down(depth_emb) # (B*horizon, out_size)
            depth_emb = depth_emb.view(batch_size, image_horizon, self.out_size) # (B, horizon, out_size)
            
            view_emb = torch.concatenate(tensors=(rgb_emb, depth_emb), dim=-1) # (B, horizon, 2*out_size=256)
            view_embs.append(view_emb)
            
        embs = torch.concatenate(tensors=view_embs, dim=-1) # (B, image_horizon, 4*out_size=512)
                
        return embs
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):   
           init.kaiming_normal_(module.weight, nonlinearity='relu')
           nn.init.zeros_(module.bias)