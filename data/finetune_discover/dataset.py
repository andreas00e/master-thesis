import os 
import h5py 
import numpy as np
import pandas as pd 
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch.utils.data import Dataset

from data.finetune_discover.utils.transforms import get_transforms


ROBOT_DICT = {
    "iiwa": 0, 
    "panda": 1, 
    "sawyer": 2, 
    "ur5e": 3 
    }

TASK_DICT = {
    "square": 0, 
    "threading": 1
}


class MimicGenRobotDataset(Dataset):
    def __init__(
        self,
        demo_map: List[Tuple[Union[str, Path], str, int]],
        dataframe_gripper: pd.DataFrame, 
        transforms_list: List[str], 
        contrastive_transforms: Optional[bool]=None, 
        window_size: Optional[int]=None,
        chunk_size: Optional[int]=None,
        temporal_smoothing: bool=False, 
        positive_window_size: Optional[int]=None, 
        negative_window_size: Optional[int]=None, 
        crop_factor: float=1.0, 
        noise_level: Optional[float]=None,
        epoch: int=1, 
        modalities: Optional[List[str]] = None,
        )-> None: 
        super().__init__()
        
        self.demo_map = demo_map
        self.dataframe_gripper = dataframe_gripper
        self.transforms_list = transforms_list
        self.contrastive_transforms = contrastive_transforms 
        self.window_size = window_size
        self.chunk_size = chunk_size
        self.temporal_smoothing = temporal_smoothing
        self.positive_window_size = positive_window_size
        self.negative_window_size = negative_window_size
        self.crop_factor = crop_factor
        self.noise_level = noise_level

        self.modalities = modalities if modalities is not None else ["rgb_one", "rgb_two", "g_qpos"]
        
        self.gripper_mins, self.gripper_maxs = {}, {}
        for robot in ROBOT_DICT.keys(): 
            min_col, max_col = f"{robot}_min", f"{robot}_max"
            if min_col in dataframe_gripper.columns and max_col in dataframe_gripper.columns: 
                self.gripper_mins[robot] = dataframe_gripper[min_col].values
                self.gripper_maxs[robot] = dataframe_gripper[max_col].values
        
        self._parsed_metadata = []
        for file_path, demo, idx in demo_map: 
            stem = Path(file_path).stem
            task = stem.split("_")[0]
            robot = stem.split("_")[-1]
            self._parsed_metadata.append({
                "file_path": file_path,
                "demo": demo,
                "task_id": TASK_DICT[task],
                "robot_id": ROBOT_DICT[robot],
                "robot_name": robot
            })
        
        self._file_cache: Dict[str, h5py.File] = {}  
        self._pid: Optional[int] = None
        
        self.epoch = epoch

    @property
    def epoch(self) -> int:
        return self._epoch

    @epoch.setter
    def epoch(self, value: int) -> None:
        self._epoch = value
        self.transforms = get_transforms(self.transforms_list, self._epoch)  
        if self.contrastive_transforms: 
            self.positive_transforms = get_transforms(self.transforms_list, self._epoch)

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
    
    def close(self) -> None:
        for hf in self._file_cache.values():
            try: 
                hf.close()
            except Exception: 
                pass  
        self._file_cache.clear()

    def __del__(self) -> None:
        self.close()
    
    def __len__(self) -> int:
        return len(self.demo_map)

    def __getitem__(self, idxs: int) -> Dict[str, torch.Tensor]: 
        meta = self._parsed_metadata[idxs]
        file_path, demo = meta["file_path"], meta["demo"]
        robot_name = meta["robot_name"]
        task_id = meta["task_id"]
        robot_id = meta["robot_id"]
        
        hf = self._get_hdf5_handle(file_path)
        demo_obs = hf["data"][demo]["obs"]
        
        n_steps = demo_obs["robot0_eye_in_hand_image"].shape[0]
    
        if self.positive_window_size is not None and self.negative_window_size is not None: # time-contrastive sampling 
            max_offset = max(1, n_steps-self.window_size+1) 
            
            start_ = torch.randint(0, max_offset, (1,)).item()
            idxs = torch.arange(start_, start_+self.window_size)
            
            start_pos = torch.randint(start_-self.positive_window_size, start_+1, (1,)).item()
            pos_idxs = torch.arange(start_pos, start_pos+self.window_size)

            start_neg = torch.randint(start_-self.negative_window_size-self.window_size, start_-self.negative_window_size+1, (1,)).item()
            neg_idxs = torch.arange(start_neg, start_neg+self.window_size)
            
            idxs = torch.vstack((idxs, pos_idxs, neg_idxs))
            idxs = torch.clamp(idxs, 0, n_steps-1)
        
        elif self.window_size is not None and self.chunk_size is not None: 
            if self.temporal_smoothing:
                max_offset = max(1, n_steps - self.window_size + 1)
                start = torch.randint(0, max_offset, (1,)).item()  
                
                idx_range = torch.arange(start=start - self.window_size, end=start + self.window_size*2)
                idx_range = torch.clamp(idx_range, min=0, max=n_steps - 1)
                idxs = torch.stack(torch.chunk(idx_range, 3), dim=0) # [3, window] 
            else: 
                max_offset = max(1, n_steps - self.window_size + 1)
                offsets = torch.randint(0, max_offset, size=(self.chunk_size, )) # indexing, not slicing
                idxs = offsets[:, None] + torch.arange(self.window_size) # [chunk, window]
        else: 
            idxs = torch.arange(n_steps) # [whole trajectory] 

        flat = idxs.flatten()
        uniq, inv = np.unique(flat, return_inverse=True)
        
        item = {}
        
        # 1. Perspective 1: robot_0_eye_in_hand_image
        if "rgb_one" in self.modalities: 
            rgb_one = demo_obs["robot0_eye_in_hand_image"][uniq] # [n_steps, height=84, width=84, channels=3]
            if self.crop_factor is not None: 
                crop_h = int(rgb_one.shape[1]*self.crop_factor)
                rgb_one = rgb_one[:, :crop_h, ...]
            rgb_one = torch.from_numpy(rgb_one).permute(0, 3, 1, 2) # [uniq, channels=3, height=224, width=224] 
        
        # 2. Perspective 2: agentview_image 
        if "rgb_two" in self.modalities: 
            rgb_two = demo_obs["agentview_image"][uniq] # [n_steps, height=84, width=84, channels=3]       
            rgb_two = torch.from_numpy(rgb_two).permute(0, 3, 1, 2) # [uniq, channels=3, height=224, width=224]       
        
        # 3. Normalized gripper joint states 
        if "g_qpos" in self.modalities: 
            g_qpos = demo_obs["robot0_gripper_qpos"][uniq] # [uniq, d]: d in {2, 6}
        
            if robot_name in self.gripper_mins:             
                g_min = self.gripper_mins[robot_name][:g_qpos.shape[-1]] # [1, d]
                g_max = self.gripper_maxs[robot_name][:g_qpos.shape[-1]] # [1, d]
                g_qpos = np.clip((g_qpos - g_min) / ((g_max - g_min) + 1e-8), 0.0, 1.0) # [uniq, d] 
            
            g_qpos = np.mean(g_qpos, axis=-1) # [uniq]
            g_qpos = g_qpos[inv].reshape(*idxs.shape, 1) # [chunk, window, 1]
            g_qpos = torch.from_numpy(g_qpos).to(torch.float32) 
            
            item["g_qpos"] = g_qpos

        # Contrastive Transforms
        if self.contrastive_transforms and hasattr(self, "positive_transforms"): 
            if "rgb_one" in self.modalities:      
                rgb_one_pos = self.positive_transforms(rgb_one) # positive sample
                rgb_one_pos = rgb_one_pos[inv].view(*idxs.shape, *rgb_one_pos.shape[1:])
                item["rgb_one_pos"] = rgb_one_pos

            if "rgb_two" in self.modalities: 
                rgb_two_pos = self.positive_transforms(rgb_two) # positive sample
                rgb_two_pos = rgb_two_pos[inv].view(*idxs.shape, *rgb_two_pos.shape[1:]) 
                item["rgb_two_pos"] = rgb_two_pos

            if "g_qpos" in self.modalities: 
                edge_damping = g_qpos * (1.0 - g_qpos) 
                noise = self.noise_level * torch.randn_like(g_qpos) * edge_damping
                g_qpos_pos = torch.clamp(g_qpos + noise, 0.0, 1.0)
            
                item["g_qpos_pos"] = g_qpos_pos
        
        # Standard Transforms
        if hasattr(self, "transforms"): 
            if "rgb_one" in self.modalities: 
                rgb_one = self.transforms(rgb_one)
                rgb_one = rgb_one[inv].view(*idxs.shape, *rgb_one.shape[1:])
                item["rgb_one"] = rgb_one
                
            if "rgb_two" in self.modalities:
                rgb_two = self.transforms(rgb_two)
                rgb_two = rgb_two[inv].view(*idxs.shape, *rgb_two.shape[1:]) 
                item["rgb_two"] = rgb_two
                        
        item["task"] = torch.full(size=(idxs.shape[0], ), fill_value=task_id, dtype=torch.long) # [chunk]
        item["robot"] = torch.full(size=(idxs.shape[0], ), fill_value=robot_id, dtype=torch.long) # [chunk]
        item["idxs"] = idxs # [chunk, window]

        return item