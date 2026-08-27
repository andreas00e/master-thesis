import os 
import h5py 
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

import torch
from torchtyping import TensorType 
from torch.utils.data import Dataset 

from data.discover.utils.transforms import get_transforms


class MimicGenRobotDataset(Dataset): 
    def __init__(self, 
        demo_map: List[Tuple[os.PathLike, int, int]],
        robot: str, 
        task: str, 
        df_gripper: pd.DataFrame, 
        transforms: List[str],
        noise_level: float, 
        crop_factor: Optional[float]=None,
        window: Optional[int]=None,
        chunk: Optional[int]=None
        ) -> None:
        super().__init__()
        
        if crop_factor and not 0 < crop_factor <= 1: raise ValueError(f"crop_factor must be in (0, 1], got {crop_factor}")
        if noise_level < 0: raise ValueError(f"noise_level must be non-negative, got {noise_level}")
        
        self.demo_map = demo_map
        self.df_gripper = df_gripper
        self.crop_factor = crop_factor
        self.noise_level = noise_level
        self.window = window
        self.chunk = chunk

        self.positive_transforms = get_transforms(transforms)
        self.transforms = get_transforms(transforms)
                
        self._file_cache: Dict[os.PathLike, h5py.File] = {}  
                             
    def __len__(self) -> int:
        return len(self.demo_map)
    
    def __getstate__(self) -> Dict:
        state = self.__dict__.copy() # when the worker is forked/pickled, clear the cache 
        state["_file_cache"] = {}
        return state
    
    def close(self) -> None:
        for hf in self._file_cache.values():
            try: 
                hf.close()
            except Exception: 
                pass  
        self._file_cache.clear()

    def __del__(self) -> None:
        self.close()

    def _get_hdf5_handle(self, file_path: os.PathLike) -> h5py.File:
        hf = self._file_cache.get(file_path)
        if hf is None:
            hf = h5py.File(file_path, "r")
            self._file_cache[file_path] = hf
        return hf
        
    def __getitem__(self, idx: int) -> Dict[str, TensorType["*"]]: 
        item = {}
        file_path, demo, _ =  self.demo_map[idx]
        
        robot = os.path.basename(file_path).split(".")[0].split("_")[-1]
    
        hf = self._get_hdf5_handle(file_path)

        # First perspective: robot_0_eye_in_hand_image
        rgb_one = hf["data"][demo]["obs"]["robot0_eye_in_hand_image"][()]
        if self.crop_factor is not None: 
            rgb_one = rgb_one[:, :int(rgb_one.shape[1]*self.crop_factor), ...]
 
        rgb_one = torch.from_numpy(rgb_one).permute(0, 3, 1, 2) # [n, c, h, w]    
        rgb_one_pos = self.positive_transforms(rgb_one) # [n, c=3, h=224, w=224]: positive sample 
        rgb_one = self.transforms(rgb_one) # [n, c=3, h=224, w=224]: anchor sample
        
        # Second perspective: agentview_image 
        rgb_two = hf["data"][demo]["obs"]["agentview_image"][()]
        rgb_two = torch.from_numpy(rgb_two).permute(0, 3, 1, 2) # [n, c, h, w]    
        rgb_two_pos = self.positive_transforms(rgb_two) # [n, c=3, h=224, w=224]
        rgb_two = self.transforms(rgb_two) # [n, c=3, h=224, w=224]  
        
        # Gripper joint states 
        g_qpos = hf["data"][demo]["obs"]["robot0_gripper_qpos"][()] # [n, d]: d in {2, 6}
        g_min = self.df_gripper[f"{robot}_min"].values[:g_qpos.shape[1]][None, :] # [1, d]
        g_max = self.df_gripper[f"{robot}_max"].values[:g_qpos.shape[1]][None, :] # [1, d]
        g_qpos = np.clip((g_qpos - g_min) / (g_max - g_min + 1e-8), 0.0, 1.0) # [n, d]
        g_qpos = np.mean(g_qpos, axis=-1)
        g_qpos = torch.from_numpy(g_qpos).float() # [n]
        g_qpos_plus = torch.clamp(g_qpos + self.noise_level * torch.randn_like(g_qpos), 0.0, 1.0) # [n]
        
        seq_len = rgb_one.shape[0]
        if self.window is not None and self.chunk is not None: 
            max_offset = max(1, seq_len - self.window + 1)
            offsets = torch.randint(0, max_offset, size=(self.chunk, )) # indexing, not slicing
            idxs = offsets[:, None] + torch.arange(self.window) # [chunk, window]
        else: 
            idxs = torch.arange(seq_len) # [whole trajectory]
        
        item["rgb_one"] = rgb_one[idxs, ...] # [chunk, window, c, h, w]
        item["rgb_one_plus"] = rgb_one_pos[idxs, ...] 
        item["rgb_two"] = rgb_two[idxs, ...]
        item["rgb_two_plus"] = rgb_two_pos[idxs, ...]
        item["g_qpos"] = g_qpos[idxs, ...]
        item["g_qpos_plus"] = g_qpos_plus[idxs, ...]
        item["idxs"] = idxs
        
        return item 