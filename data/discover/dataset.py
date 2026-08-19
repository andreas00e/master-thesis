import os 
import h5py 
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

import torch
from torchtyping import TensorType 
from torch.utils.data import Dataset 

from data.discover.utils.transforms import get_transforms


class MimicGenRobotDataset(Dataset): 
    def __init__(self, 
        demo_map: List[Tuple[os.PathLike, int, int]],
        df_g: pd.DataFrame, 
        window: int,
        chunks: int, 
        crop_factor: float, 
        noise_level: float, 
        transforms: List[str]
        ) -> None:
        super().__init__()
        
        assert 0 < crop_factor <= 1, ValueError(f"crop_factor must be in (0, 1], got {crop_factor}")
        assert noise_level > 0, ValueError(f"noise_level must be non-negative, got {noise_level}")
        
        self.demo_map = demo_map
        self.df_g = df_g
        self.window = window
        self.chunks = chunks 
        self.crop_factor = crop_factor
        self.noise_level = noise_level

        self.transforms_pos = get_transforms(transforms)
        self.transforms = get_transforms(transforms)
                
        self._file_cache = {}
                       
    def __len__(self) -> int:
        return len(self.demo_map)
    
    def __getstate__(self) -> Dict:
        state = self.__dict__.copy() # when the worker is forked/pickled, clear the cache 
        state["_file_cache"] = {}
        return state
    
    def close(self) -> None:
        for hf in self._file_cache.values():
            hf.close()
        self._file_cache.clear()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def __getitem__(self, idx: int) -> Dict[str, TensorType["*"]]: 
        item = {}
        file, demo, _ =  self.demo_map[idx]
        
        robot = file.split(".")[0].split("_")[-1]

        hf = self._file_cache.get(file)
        if hf is None:
            hf = h5py.File(file, "r")
            self._file_cache[file] = hf

        # First perspective: robot_0_eye_in_hand_image
        rgb_one = hf["data"][demo]["obs"]["robot0_eye_in_hand_image"]
        rgb_one = rgb_one[:, :int(rgb_one.shape[1]*self.crop_factor), ...]
        rgb_one = torch.from_numpy(rgb_one).permute(0, 3, 1, 2) # [n, c, h, w]    
        rgb_one_pos = self.transforms_pos(rgb_one) # [n, c=3, h=224, w=224]: positive sample 
        rgb_one = self.transforms(rgb_one) # [n, c=3, h=224, w=224]: anchor sample
        
        # Second perspective: agentview_image 
        rgb_two = hf["data"][demo]["obs"]["agentview_image"][()]
        rgb_two = torch.from_numpy(rgb_two).permute(0, 3, 1, 2) # [n, c, h, w]    
        rgb_two_pos = self.transforms_pos(rgb_two) # [n, c=3, h=224, w=224]
        rgb_two = self.transforms(rgb_two) # [n, c=3, h=224, w=224]  
            
        offsets = torch.randint(0, rgb_one.shape[0]-self.window+1, size=(self.chunks, )) # indexing, not slicing
        idxs = offsets[:, None] + torch.arange(self.window) # [chunks, window]
       
        g_qpos_full = hf["data"][demo]["obs"]["robot0_gripper_qpos"][()] # [n, d]: d in {2, 6}
        g_min = self.df_g[f"{robot}_min"].values[:g_qpos_full.shape[1]][None, :] # [1, d]
        g_max = self.df_g[f"{robot}_max"].values[:g_qpos_full.shape[1]][None, :] # [1, d]
        
        g_qpos_full = np.clip((g_qpos_full - g_min) / (g_max - g_min), 0.0, 1.0) # [n, d]
        g_qpos_full = np.mean(g_qpos_full, axis=-1)
        g_qpos_full = torch.from_numpy(g_qpos_full).to(torch.float32) # [n]
        g_qpos = g_qpos_full[idxs] # [chunks, window]
        g_qpos_plus = torch.clip(g_qpos + self.noise_level * torch.randn_like(g_qpos), 0.0, 1.0)
        

        
        item["rgb_one"] = rgb_one[idxs, ...]
        item["rgb_one_plus"] = rgb_one_pos[idxs, ...] 
        item["rgb_two"] = rgb_two[idxs, ...]
        item["rgb_two_plus"] = rgb_two_pos[idxs, ...]
        item["g_qpos"] = g_qpos[idxs]
        item["g_qpos_plus"] = g_qpos_plus[idxs]
        item["idxs"] = idxs
        return item 