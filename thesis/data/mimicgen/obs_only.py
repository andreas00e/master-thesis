import os 
import h5py 
import numpy as np 
from typing import List, Optional 

import torch 
from torch.utils.data import Dataset 
from torchvision import transforms

os.environ["PYTHONHASSEED"] = "0"


class MimicgenDataset(Dataset): 
    def __init__(self, file_dir: os.PathLike, horizon: Optional[int]=16, robots: Optional[List[str]]=None, tasks: Optional[List[str]]=None, depth: Optional[bool]=False, expand_depth: Optional[str]=None) -> None:
        super().__init__()
        
        self.file_dir = file_dir
        self.depth = depth
        all_files = [file for file in os.listdir(self.file_dir) if ("depth" in file) == self.depth]
        
        all_robots = list(set(f.split(".")[-2].split("_")[-1] for f in all_files)) # no given robot -> all robots
        all_tasks = list(set(f.split(".")[-2].split("_")[0] for f in all_files)) # no given task -> all tasks

        self.robots = robots if robots else all_robots
        self.tasks = tasks if tasks else all_tasks 
        
        files = [file for file in all_files if (robot in file for robot in self.robots) and (task in file for task in self.tasks)]
        
        self.horizon = horizon
        self.expand_detph = expand_depth # grayscale, colormap 
        
        self.len = 0
        self._trajectory_map, self._rgb_views, self._depth_views = [], [], []         
        self._file_cache = {}
        
        for file in files: 
            with h5py.File(os.path.join(self.file_dir, file), "r") as hf: 
                data = hf["data"]
                for demo in data.keys(): 
                    n_steps = data[demo]["actions"][()].shape[0]
                    self.len += n_steps
                    self._trajectory_map.append([file, demo, n_steps])

    def __len__(self) -> int:
        return self.len 
    
    def __getstate__(self):
        state = self.__dict__.copy() # when the worker is forked/pickled, clear the cache 
        state["_file_cache"] = {}
        
        return state
    
    def __getitem__(self, idx): 
        file, demo, n_steps = self._trajectory_map[idx] 
        try:  
            start_idx = np.random.randint(0, n_steps-self.horizon+1) # [0, n_steps-self.horizon]
        except: 
            raise ValueError
        
        return self.load_trajectory(file, demo, start_idx)
    
    def load_trajectory(self, file: str, demo: int, start_idx: int):     
        out = {}    
        
        if file not in self._file_cache: # open file once per worker and cache it
            self._file_cache[file] = h5py.File(os.path.join(self.file_dir, file), "r")
            
        hf = self._file_cache[file]
        data = hf["data"][demo]
        obs = data["obs"]    
        
        out["file"] = "-".join(file.split(".")[0].split("_")[-1::-2])

        horizon = range(start_idx, start_idx+self.horizon)

        rgb_obs = obs["robot0_eye_in_hand_image"][[*horizon], ...] # [horizon, H, W, 3]
        rgb_obs = rgb_obs.to(np.float32)
        rgb_obs /= 255.0
        out["rgb_obs"] = rgb_obs
        
        if self.depth: 
            d_obs = obs["robot0_eye_in_hand_depth"][[*horizon], ...] # [horizon, H, W, 1]
            out["d_obs"] = d_obs
            
        return out