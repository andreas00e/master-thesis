import os 
import cv2
import h5py
import numpy as np 
from typing import List, Union 

import torch
from torch.utils.data import Dataset


class MimicgenDataset(Dataset):
    def __init__(self, file_dir: os.PathLike, robots: Union[List[str], str], tasks: Union[List[str], str], action_horizon: int, image_horizon: int, expand_depth: str):
        super().__init__()
        
        self.file_dir = file_dir
        self.all_files = [file for file in os.listdir(self.file_dir) if file.endswith("depth.hdf5")]
        
        self.all_robots = list(set(f.split(".")[-2].split("_")[-1] for f in self.all_files))
        self.all_tasks = list(set(f.split(".")[-2].split("_")[0] for f in self.all_files))
        
        self.robots = robots if robots else self.all_robots # no given robot -> all robots
        self.tasks = tasks if tasks else self.all_tasks # no given task -> all tasks
        
        self.files = [os.path.join(self.file_dir, file) for file in self.all_files if (robot in file for robot in self.robots) and (task in file for task in self.tasks)]

        self.action_horizon = action_horizon
        self.image_horizon = image_horizon 
        self.expand_depth = expand_depth # grayscale, colormap
        
        self.trajectory_map, self.rgb_views, self.depth_views = [], [], []
        
        for file in self.files: 
            with h5py.File(file, "r") as hf: 
                data = hf["data"]
                for demo in data.keys(): 
                    n_actions = data[demo]["actions"][()].shape[0]
                    self.trajectory_map.extend((file, demo, idx) for idx in range(n_actions))
                             
    def __len__(self) -> int: # TODO: Check exact type that is being returned!
        return len(self.trajectory_map)          

    def __getitem__(self, idx: int):
        input = self.trajectory_map[idx] # -> (file, demo, index)
        return self.load_trajectory(*input) 
    
    def load_trajectory(self, file: str, demo: int, idx: int): # "Actions should be normalized between -1 and 1": https://robomimic.github.io/docs/datasets/overview.html
        observations = {}
        
        with h5py.File(file, "r") as hf:
            data = hf["data"][demo] 
            actions = data["actions"][()] # EEF ACTION: [n_steps, action_dim]
            obs = data["next_obs"] # TASK OBSERVATIONS (containing all available RGB views and depth maps)    
             
            n_steps = actions.shape[0]      

            if idx+self.action_horizon <= n_steps: # idx + horizon <= number of action steps in demo 
                actions = actions[idx:idx+self.action_horizon, :]       
            else: # idx + horizon > number of action steps in demo => padding
                actions_overhead = (idx+self.action_horizon)-actions.shape[0] # overhead = number of missing/padded steps
                actions_padding = np.zeros((actions_overhead, actions.shape[1])) 
                actions = np.vstack((actions[idx:, :], actions_padding))      
                
            actions = torch.from_numpy(np.float32(actions))
            
            for rgb_view, depth_view in zip(self.rgb_views, self.depth_views): # "agentview", "robot0_eye_in_hand"
                rgb_images = obs[rgb_view][()] # rgb view observations [n_steps, H, W, C=3]
                depth_images = obs[depth_view][()] # depth view observations [n_steps, H, W, C=1]
                    
                frames = np.concatenate([rgb_images, depth_images],  axis=-1) # => [n_steps, H, W, C=4]
                
                if self.image_horizon == 1: 
                    frames = frames[idx, ...].unsqueeze(0) # one image per step
                elif self.image_horizon < 0: 
                    if idx >= abs(self.image_horizon): 
                        frames = frames[idx+self.image_horizon, ...] # [image_horizon, H, W, C]
                    else:  
                        frames = frames[:idx, ...] # [idx, H, W, X]  
                else: # self.n_obs == 0: 
                    raise ValueError("Not forwarding any image data is not supported!")
                   
                min_depth = np.min(depth_images)
                max_depth = np.max(depth_images)
                    
                frames = np.transpose(a=frames, axes=(0, 3, 1, 2)) # [horizon, H, W, C=6] -> [horizon, C=6, H, W]
                frames[:, 3, ...] = (frames[:, 3, ...]-min_depth)/(max_depth-min_depth) # Normaize depth map to [0, 1]
                frames[:, 3, ...] *= 255. 
                 
                if self.expand_depth: 
                    new_frames = np.zeros((frames.shape[0], 6, *frames.shape[-2:]))
                    new_frames[:, :3, ...] = frames[:, :3, ...] # copy stacked observations rgb part
                    
                    if self.expand_depth == "colormap": # gray scale image -> colorized image w/ colormap
                        depth_frames = np.stack([cv2.cvtColor(cv2.applyColorMap(frames[i, 3, ...].astype(np.uint8), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB) for i in range(frames.shape[0])])
                    elif self.expand_depth == "grayscale": # repeat depth channel three times
                        depth_frames = frames[:, 3:3+1, ...]
                        depth_frames = np.repeat(depth_frames, repeats=3, axis=1) # [horizon, C=3, H, W]
                        
                    depth_frames = np.transpose(depth_frames, axes=(0, 3, 1, 2)) # [horizon, C=3, H, W] -> [horizon, H, W, C=3]
                    new_frames[:, 3:, ...] = depth_frames
                    frames = new_frames

                frames = torch.from_numpy(np.float32(frames))
                
                view = "_".join(rgb_view.split("_")[:-1])
                observations[view] = frames 
            
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            # ROTATE BOTH IMAGES IN thesis/robosuite/run.py
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            
            # rand_int = random.randint(a=0, b=len(self.instructions)-1) # same demo gets different language instruction 
            i_idx = int(demo.split("_")[-1]) % len(self.instructions) # same demo always gets same language instruction
            instructions = self.instructions[i_idx] # LANGUAGE INSTRUCTION 

        return {
            "actions": actions, # EEF ACTIONS       
            "obs": observations, # TASK OBSERVATIONS (dict)
            "text": instructions # LANGUAGE INSTRUCTION 
        }