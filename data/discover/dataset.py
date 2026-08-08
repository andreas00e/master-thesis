import os 
import h5py 
import pandas as pd
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset 

from data.discover.utils.transforms import get_transforms


class MimicGenRobotDataset(Dataset): 
    def __init__(self, 
        demo_map: List[Tuple[os.PathLike, int, int]],
        df_gripper: pd.DataFrame, 
        depths: pd.DataFrame,
        window: int,
        chunks: int, 
        crop_factor: float, 
        depth: Optional[bool], 
        expand_depth: str, # grayscale, colormap 
        transforms: List[str]
        ) -> None:
        super().__init__()
        
        self.demo_map = demo_map
        self.df_gripper = df_gripper
        self.depths = depths
        self.window = window
        self.chunks = chunks 
        self.crop_factor = crop_factor
        self.depth = depth 
        self.expand_depth = expand_depth 
        self.transforms = get_transforms(transforms)
        self.transforms_plus = get_transforms(transforms)
        
        self._file_cache = {}
                       
    def __len__(self) -> int:
        return len(self.demo_map)
    
    def __getstate__(self) -> Dict:
        state = self.__dict__.copy() # when the worker is forked/pickled, clear the cache 
        state["_file_cache"] = {}
        return state
    
    # def __del__(self):
    #   for f in self._file_cache.values():
    #       try:
    #           f.close()
    #       except Exception:
    #           pass
    
    def __getitem__(self, idx):
        item = {}
        file, demo, n_steps =  self.demo_map[idx]
        
        robot = file.split(".")[-1]
        task = file.split(".")[-1]
        
        offset = torch.randint(low=0, high=n_steps-self.window, shape=(self.chunks, ))
        idxs = torch.arange(start=0, end=self.window) + offset

        hf = self._file_cache.get(file)
        if hf is None:
            hf = h5py.File(file, "r")
            self._file_cache[file] = hf

        rgb_obs = hf["data"][demo]["obs"]["robot0_eye_in_hand_image"]
        rgb_obs = rgb_obs[:, :int(rgb_obs.shape[1]*self.crop_factor), ...]
        rgb_obs = torch.from_numpy(rgb_obs).permute(0, 3, 1, 2) # [n, c, h, w]
                        
        rgb = self.transforms(rgb_obs) 
        rgb_plus = self.transforms(rgb_obs)
        
        item["rgb"] = rgb[idxs]
        item["rgb_plus"] = rgb_plus[idxs] 
        
        
        if self.depth: 
            depth_obs = hf["data"][demo]["obs"]["robot0_eye_in_hand_depth"]
            depth_obs = depth_obs[:, :int(depth_obs.shape[1]*self.crop_factor), ...]
            depth_obs = torch.from_numpy(depth_obs).permute(0, 3, 1, 2) # [n, 1, h, w]
            
            depth = self.transforms(depth_obs) 
            depth_plus = self.transforms(depth_obs)
            item["depth"] = depth[idxs]
            item["depth_plus"] = depth_plus[idxs]
            
        gripper_qpos = hf["data"][demo]["obs"]["robot0_gripper_qpos"] # [n, d]: d in {2, 6}
        gripper_min = self.df_gripper[robot][task]["min"] # [d, ]
        gripper_max = self.df_gripper[robot][task]["max"] # [d, ]
        
        gripper_qpos[:, 0] = (gripper_qpos[:, 0] - gripper_min[0]) / (gripper_max[0] - gripper_min[0]) 
        gripper_qpos[:, 1] = (gripper_qpos[:, 1] - gripper_min[1]) / (gripper_max[1] - gripper_min[1]) 
        
        item["gripper_qpos"] = gripper_qpos[idxs]
        
        return item 