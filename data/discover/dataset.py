import os 
import h5py 
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch.utils.data import Dataset 

from data.discover.utils.transforms import get_transforms

TASK_DICT = {
    "square": 0, 
    "threading": 1
}

ROBOT_DICT = {
    "iiwa": 0, 
    "panda": 1, 
    "sawyer": 2, 
    "ur5e": 3 
}

class MimicGenRobotDataset(Dataset): 
    def __init__(self, 
        demo_map: List[Tuple[Union[str, Path], str, int]],
        df_gripper: pd.DataFrame, 
        window: int,
        chunk: int,
        crop_factor: Optional[float], 
        noise_level: float,
        ctr_transforms: bool,  
        transforms: List[str],
        ) -> None:
        super().__init__()
        
                
        self.demo_map = demo_map
        self.df_gripper = df_gripper
        self.window = window
        self.chunk = chunk
        self.crop_factor = crop_factor
        self.noise_level = noise_level
        
        if ctr_transforms: 
            self.anchor_transforms = get_transforms(transforms)
            self.positive_transforms = get_transforms(transforms)
        else: 
            self.transforms = get_transforms(transforms)
                
        self._file_cache: Dict[str, h5py.File] = {}  
        self._pid: Optional[int] = None
                             
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

    def _get_hdf5_handle(self, file_path: Union[str, os.PathLike]) -> h5py.File:
        current_pid = os.getpid() 
        
        if self._pid != current_pid: 
            self._file_cache.clear()
            self._pid = current_pid
        
        path_str = str(file_path)
        hf = self._file_cache.get(path_str)
        if hf is None:
            hf = h5py.File(path_str, "r")
            self._file_cache[path_str] = hf
        return hf
        
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]: 
        item = {}
        file_path, demo, _ =  self.demo_map[idx]
        
        task = Path(file_path).stem.split("_")[0]
        robot = Path(file_path).stem.split("_")[-1]
        
        hf = self._get_hdf5_handle(file_path)
        demo_obs = hf["data"][demo]["obs"]
        
        n_steps = demo_obs["robot0_eye_in_hand_image"].shape[0]
        
        if self.window is not None and self.chunk is not None: 
            max_offset = max(1, n_steps - self.window + 1)
            offsets = torch.randint(0, max_offset, size=(self.chunk, )) # indexing, not slicing
            idxs = offsets[:, None] + torch.arange(self.window) # [chunk, window]
        else: 
            idxs = torch.arange(n_steps) # [whole trajectory] 
 
        flat = idxs.flatten()
        uniq, inv = np.unique(flat, return_inverse=True)
        
        # 1. Perspective 1: robot_0_eye_in_hand_image
        rgb_one = demo_obs["robot0_eye_in_hand_image"][uniq]
        if self.crop_factor is not None: 
            crop_h = int(rgb_one.shape[1]*self.crop_factor)
            rgb_one = rgb_one[:, :crop_h, ...]
        rgb_one = torch.from_numpy(rgb_one).permute(0, 3, 1, 2) # [uniq, channels=3, height=224, width=224] 
        
        # 2. Perspective 2: agentview_image 
        rgb_two = demo_obs["agentview_image"][uniq]        
        rgb_two = torch.from_numpy(rgb_two).permute(0, 3, 1, 2) # [uniq, channels=3, height=224, width=224]       
        
        # 3. Normalized gripper joint states 
        g_qpos = demo_obs["robot0_gripper_qpos"][uniq] # [uniq, d]: d in {2, 6}
        min_col, max_col = f"{robot}_min", f"{robot}_max"
        
        if min_col in self.df_gripper.columns and max_col in self.df_gripper.columns: 
            g_min = self.df_gripper[min_col].values[:g_qpos.shape[-1]] # [1, d]
            g_max = self.df_gripper[max_col].values[:g_qpos.shape[-1]] # [1, d]
            g_qpos = np.clip((g_qpos - g_min) / ((g_max - g_min) + 1e-8), 0.0, 1.0) # [uniq, d] 
            
        g_qpos = np.mean(g_qpos, axis=-1) # [uniq]
        g_qpos = g_qpos[inv].reshape(*idxs.shape, 1) # [chunk, window, 1] or 
        g_qpos = torch.from_numpy(g_qpos).float()
        
        if hasattr(self, "anchor_transforms") and hasattr(self, "positive_transforms"): 
            rgb_one_pos = self.positive_transforms(rgb_one) # positive sample
            rgb_one = self.anchor_transforms(rgb_one) # anchor sample
            
            rgb_one = rgb_one[inv].view(*idxs.shape, *rgb_one.shape[1:]) 
            rgb_one_pos = rgb_one_pos[inv].view(*idxs.shape, *rgb_one_pos.shape[1:])

            rgb_two_pos = self.positive_transforms(rgb_two) # positive sample
            rgb_two = self.anchor_transforms(rgb_two) # anchor sample

            rgb_two = rgb_two[inv].view(*idxs.shape, *rgb_two.shape[1:])
            rgb_two_pos = rgb_two_pos[inv].view(*idxs.shape, *rgb_two_pos.shape[1:]) 
            
            g_qpos_plus = torch.clamp(g_qpos + self.noise_level * torch.randn_like(g_qpos), 0.0, 1.0) # positive sample
            
            item["rgb_one_pos"] = rgb_one_pos
            item["rgb_two_pos"] = rgb_two_pos
            item["g_qpos_plus"] = g_qpos_plus

        elif hasattr(self, "transforms"): 
            rgb_one = self.transforms(rgb_one)
            rgb_one = rgb_one[inv].view(*idxs.shape, *rgb_one.shape[1:])
            
            rgb_two = self.transforms(rgb_two)
            rgb_two = rgb_two[inv].view(*idxs.shape, *rgb_two.shape[1:])
            
        item["rgb_one"] = rgb_one
        item["rgb_two"] = rgb_two
        item["g_qpos"] = g_qpos
        
        item["task"] = torch.full(size=(idxs.shape[0], ), fill_value=TASK_DICT[task], dtype=torch.long) # [chunk]
        item["robot"] = torch.full(size=(idxs.shape[0], ), fill_value=ROBOT_DICT[robot], dtype=torch.long) # [chunk]
        item["idxs"] = idxs

        return item 