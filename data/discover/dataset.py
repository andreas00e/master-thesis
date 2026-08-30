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
        transforms: List[str],
        ) -> None:
        super().__init__()
                
        self.demo_map = demo_map
        self.df_gripper = df_gripper
        self.window = window
        self.chunk = chunk
        self.crop_factor = crop_factor
        self.noise_level = noise_level
        self.transforms = transforms

        self.anchor_transforms = get_transforms(self.transforms)
        self.positive_transforms = get_transforms(self.transforms)
                
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
        idxs = idxs.numpy()
        
        # 1. Perspective 1: robot_0_eye_in_hand_image
        rgb_one = demo_obs["robot0_eye_in_hand_image"][idxs]
        if self.crop_factor is not None: 
            crop_h = int(rgb_one.shape[1]*self.crop_factor)
            rgb_one = rgb_one[:, :crop_h, ...]
 
        rgb_one = torch.from_numpy(rgb_one).permute(0, 3, 1, 2) # [n, c, h, w]    
        rgb_one_pos = self.positive_transforms(rgb_one) # [n, c=3, h=224, w=224]: positive sample 
        rgb_one_anc = self.anchor_transforms(rgb_one) # [n, c=3, h=224, w=224]: anchor sample
        
        # 2. Perspective 2: agentview_image 
        rgb_two = demo_obs["agentview_image"][idxs]
        rgb_two = torch.from_numpy(rgb_two).permute(0, 3, 1, 2) # [n, c, h, w]    
        rgb_two_pos = self.positive_transforms(rgb_two) # [n, c=3, h=224, w=224]
        rgb_two_anc = self.anchor_transforms(rgb_two) # [n, c=3, h=224, w=224]  
        
        # 3. Normalized gripper joint states 
        g_qpos = demo_obs["robot0_gripper_qpos"][idxs] # [n, d]: d in {2, 6}
        min_col, max_col = f"{robot}_min", f"{robot}_max"
        
        if min_col in self.df_gripper.columns and max_col in self.df_gripper.columns: 
            g_min = self.df_gripper[min_col].values[:g_qpos.shape[1]][None, :] # [1, d]
            g_max = self.df_gripper[max_col].values[:g_qpos.shape[1]][None, :] # [1, d]
            g_qpos = np.clip((g_qpos - g_min) / (g_max - g_min + 1e-8), 0.0, 1.0) # [n, d]
            
        g_qpos = np.mean(g_qpos, axis=-1)
        g_qpos_anc = torch.from_numpy(g_qpos).float() # [n]
        g_qpos_plus = torch.clamp(g_qpos_anc + self.noise_level * torch.randn_like(g_qpos_anc), 0.0, 1.0) # [n]
        
        item["task"] = torch.tensor(TASK_DICT[task], dtype=torch.long)
        item["robot"] = torch.tensor(ROBOT_DICT[robot], dtype=torch.long)
        item["rgb_one_anc"] = rgb_one_anc # [chunk, window, c, h, w]
        item["rgb_one_plus"] = rgb_one_pos
        item["rgb_two_anc"] = rgb_two_anc
        item["rgb_two_plus"] = rgb_two_pos
        item["g_qpos_anc"] = g_qpos_anc
        item["g_qpos_plus"] = g_qpos_plus
        item["idxs"] = idxs
        
        return item 