import os 
import cv2
import h5py 
import pandas as pd
import numpy as np 
from typing import Dict, List, Optional, Union

from torch.utils.data import Dataset 

from data.utility.transforms import get_transforms


class MimicGenRobotDataset(Dataset): 
    def __init__(self, 
        demo_map: List[List[Union[os.PathLike, int, int]]],
        depths: pd.DataFrame,
        window: Optional[int],
        expand_depth: str, # grayscale, colormap 
        transforms: List[str],
        *args, **kwargs) -> None:
        super().__init__()
        
        self.demo_map = demo_map
        self.depths = depths
        self.window = window
        self.expand_detph = expand_depth 
        self.transforms = get_transforms(transforms)
        self.transforms_plus = get_transforms(transforms)
        
        self.file_chache = {}
                       
    def __len__(self) -> int:
        return len(self.demo_map)
    
    def __getstate__(self) -> Dict:
        state = self.__dict__.copy() # when the worker is forked/pickled, clear the cache 
        state["_file_cache"] = {}
        return state
    
    def __getitem__(self, idx) -> Dict: 
        item = {}  
        file, demo, n_steps = self.demo_map[idx] 
        task = file.split(".")[0].split("/")[-1].replace("_depth", "")
        print(task)
        exit() 
        idx_start = np.random.randint(0, n_steps-self.window)
        
        if file not in self.file_chache: # open file once per worker and cache it
            self.file_chache[file] = h5py.File(file, "r")
            
        hf = self.file_chache[file]
        data = hf["data"][demo]
        obs = data["obs"] 
        
        rgb_obs = obs["robot0_eye_in_hand_image"][idx_start:idx_start+self.window, ...]
        rgb_obs = rgb_obs.astype(np.float32) / 255.0
        rgb_obs = self.transforms(rgb_obs)
        rgb_obs_plus = self.transforms_plus(rgb_obs)
        item["rgb_obs"] = rgb_obs
        item["rgb_obs_plus"] = rgb_obs_plus
        
        # depth_obs = obs["robot0_eye_in_hand_depth"][idx_start:idx_start+self.window, ...] # [horizon, widht,]
        # depth_obs = np.stack([cv2.cvtColor(cv2.applyColorMap(depth_obs[i, 3, ...].astype(np.uint8), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB) for i in range(depth_obs.shape[0])])
        # item["depth_obs"] = depth_obs 
                              
        item["file"] = "_".join(file.split(".")[0].split("_")[-1::-2])+"_"+str(idx)
        return item