import os 
import h5py
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch.utils.data import Dataset

from data.finetune_discover.utils.robot_task_dict import ROBOT_DICT, TASK_DICT
from data.finetune_discover.utils.transforms import get_transforms


class MimicGenRobotDataset(Dataset):
    def __init__(
        self,
        demo_map: List[Tuple[Union[str, Path], str, int]],
        task: str, 
        robot: str, 
        gripper_mins: Optional[np.ndarray], 
        gripper_maxs: Optional[np.ndarray], 
        modalities: List[str],
        time_loss: Optional[str], 
        transforms_one_list: Optional[List[str]]=None,  
        transforms_two_list: Optional[List[str]]=None,  
        contrastive_transforms: bool=False, 
        window_size: Optional[int]=None,
        chunk_size: Optional[int]=None,
        positive_window_size: Optional[int]=None, 
        negative_window_size: Optional[int]=None, 
        crop_factor: float=1.0, 
        noise_level: Optional[float]=None,
        stage: str="fit",
        seed: int=42,
        epoch: int=1 
        )-> None: 
        super().__init__()
        
        self.demo_map = demo_map
        self.task=task
        self.robot=robot
        self.modalities = modalities
        self.time_loss = time_loss
        self.transforms_one_list = transforms_one_list 
        self.transforms_two_list = transforms_two_list       
        self.contrastive_transforms = contrastive_transforms 
        self.window_size = window_size
        self.chunk_size = chunk_size
        self.positive_window_size = positive_window_size
        self.negative_window_size = negative_window_size
        self.crop_factor = crop_factor
        self.noise_level = noise_level
        self.stage = stage
        self.seed = seed
        self.epoch = epoch 

        self.gripper_mins = gripper_mins if "g_qpos" in self.modalities else None
        self.gripper_maxs = gripper_maxs if "g_qpos" in self.modalities else None
        
        self._file_cache: Dict[str, h5py.File] = {}  
        self._pid: Optional[int] = None

    @property
    def epoch(self) -> int:
        return self._epoch

    @epoch.setter
    def epoch(self, value: int) -> None:
        self._epoch = value
                
        if "rgb_one" in self.modalities: 
            self.transforms_one = get_transforms(self.transforms_one_list, self._epoch) if self.stage in ["fit", "validate"] else get_transforms(["to_image", "to_dtype_cnn"])
        if "rgb_two" in self.modalities: 
            self.transforms_two = get_transforms(self.transforms_two_list, self._epoch) if self.stage in ["fit", "validate"] else get_transforms(["to_image", "to_dtype_cnn"])
        
        if self.contrastive_transforms: 
            if "rgb_one" in self.modalities:
                self.positive_transforms_one = get_transforms(self.transforms_one_list, self._epoch)  if self.stage in ["fit", "validate"] else get_transforms(["to_image", "to_dtype_cnn"])
            if "rgb_two" in self.modalities:
                self.positive_transforms_two = get_transforms(self.transforms_two_list, self._epoch)  if self.stage in ["fit", "validate"] else get_transforms(["to_image", "to_dtype_cnn"])

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
        if getattr(self, "_file_cache", None): 
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
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]: 
        generator = None if self.stage == "fit" else torch.Generator().manual_seed(self.seed + idx)
        sign = torch.randint(0, 2, (1, ), generator=generator).item() * 2 - 1

        file_path, demo, _ = self.demo_map[idx]
        hf = self._get_hdf5_handle(file_path)
        demo_obs = hf["data"][demo]["obs"]
        
        n_steps = demo_obs["robot0_eye_in_hand_image"].shape[0]
  
        # Sampling for time-contrastive loss             
        if self.time_loss == "contrastive" and self.positive_window_size is not None and self.negative_window_size is not None:
            max_start = n_steps - self.window_size
            
            start_ = torch.randint(0, max_start + 1, (1, ), generator=generator).item() # start index for anchor window 
            idxs = torch.arange(start_, start_ + self.window_size) # anchor window 
            
            start_pos = torch.randint(max(0, start_ - self.positive_window_size), min(start_ + self.positive_window_size, max_start) + 1, (1, ), generator=generator).item()
            
            if start_pos == start_: 
                if start_ == 0: 
                    start_pos += 1 
                elif start_ == max_start: 
                    start_pos -= 1 
                else: 
                    start_pos += sign * torch.randint(1, self.positive_window_size, (1, ), generator=generator).item()  
            
            pos_idxs = torch.arange(start_pos, start_pos + self.window_size) # positive window 
            
            list_neg = list(range(0, start_ - self.negative_window_size)) + list(range(start_ + self.negative_window_size, max_start))
            neg_idx = torch.randint(0, len(list_neg), (1, ), generator=generator).item() 
            start_neg = list_neg[neg_idx]
            neg_idxs = torch.arange(start_neg, start_neg + self.window_size) # negative window 
            
            idxs = torch.vstack((idxs, pos_idxs, neg_idxs)) # [3, self.window_size]
        
        # Sampling for time smoothing loss 
        elif self.time_loss == "smoothing" and self.window_size is not None and self.chunk_size is not None: 
            if self.temporal_smoothing:
                pad = self.window_size // 4
                max_start = max(1, n_steps - self.window_size - 2 * pad + 1)
                start = torch.randint(0, max_start, (1,), generator=generator).item() + pad

                idx_range = torch.arange(start - pad, start + self.window_size + pad)  
                idxs = idx_range.unfold(0, self.window_size, pad) # [3, window]
       
        # Training with no time loss 
        elif self.time_loss == "no_time_loss":
            max_start = max(1, n_steps - self.window_size+1)
            offsets = torch.randint(0, max_start, size=(self.chunk_size, ), generator=generator) # indexing, not slicing
            idxs = offsets[:, None] + torch.arange(self.window_size) # [chunk, window]
        
         # Loading the whole trajectory (inference)
        else: 
            idxs = torch.arange(n_steps)
        
        
        
        flat = idxs.flatten()
        uniq, inv = np.unique(flat, return_inverse=True)
        
        item = {}
        
        # 1. Perspective 1: robot_0_eye_in_hand_image
        if "rgb_one" in self.modalities: 
            rgb_one = demo_obs["robot0_eye_in_hand_image"][uniq] # [n_steps, height=84, width=84, channels=3]
            if self.crop_factor is not None: 
                crop_h = int(rgb_one.shape[1]*self.crop_factor)
                rgb_one = rgb_one[:, :crop_h, ...]
            rgb_one = torch.from_numpy(rgb_one).permute(0, 3, 1, 2) # [uniq, channels=3, height=84, width=84] 
            rgb_one = rgb_one[inv].view(*idxs.shape, *rgb_one.shape[1:]) # [chunk, window, channels=3, height=84, width=84]

        # 2. Perspective 2: agentview_image 
        if "rgb_two" in self.modalities: 
            rgb_two = demo_obs["agentview_image"][uniq] # [n_steps, height=84, width=84, channels=3]       
            rgb_two = torch.from_numpy(rgb_two).permute(0, 3, 1, 2) # [uniq, channels=3, height=84, width=84]   
            rgb_two = rgb_two[inv].view(*idxs.shape, *rgb_two.shape[1:]) # [chunk, window, channels=3, height=84, width=84]
        
        # 3. Normalized gripper joint states 
        if "g_qpos" in self.modalities: 
            g_qpos = demo_obs["robot0_gripper_qpos"][uniq] # [uniq, d]: d in {2, 6}
        
            g_min = self.gripper_mins[:g_qpos.shape[-1]] # [1, d]
            g_max = self.gripper_maxs[:g_qpos.shape[-1]] # [1, d]
            g_qpos = np.clip((g_qpos - g_min) / ((g_max - g_min) + 1e-8), 0.0, 1.0) # [uniq, d] 
            
            g_qpos = np.mean(g_qpos, axis=-1) # [uniq]
            g_qpos = g_qpos[inv].reshape(*idxs.shape, 1) # [chunk, window, 1]
            g_qpos = torch.from_numpy(g_qpos).to(torch.float32) # [chunk, window, 1]
                        
            item["g_qpos"] = g_qpos # [chunk, window, 1]

        # Contrastive Transforms
        if self.contrastive_transforms: 
            if "rgb_one" in self.modalities and self.positive_transforms_one is not None:    
                rgb_one_pos = self.positive_transforms_one(rgb_one) # positive sample with independent transformations per chunk
                item["rgb_one_pos"] = rgb_one_pos # [chunk, window, channels=3, height=84, width=84]

            if "rgb_two" in self.modalities and self.positive_transforms_two is not None: 
                rgb_two_pos = self.positive_transforms_two(rgb_two) # positive sample with independent transformations per chunk
                item["rgb_two_pos"] = rgb_two_pos # [chunk, window, channels=3, height=84, width=84]

            if "g_qpos" in self.modalities: 
                edge_damping = g_qpos * (1.0 - g_qpos) 
                noise = self.noise_level * torch.randn(g_qpos.shape, generator=generator, dtype=g_qpos.dtype) * edge_damping         
                g_qpos_pos = torch.clamp(g_qpos + noise, 0.0, 1.0)
            
                item["g_qpos_pos"] = g_qpos_pos # [chunk, window, 1] 
                
        # Standard Transforms
        if "rgb_one" in self.modalities: 
            rgb_one = self.transforms_one(rgb_one) # anchor sample with independent transformations per chunk
            item["rgb_one"] = rgb_one # [chunk, window, channels=3, height=84, width=84]
            
        if "rgb_two" in self.modalities:
            rgb_two = self.transforms_two(rgb_two) # anchor sample with independent transformations per chunk
            item["rgb_two"] = rgb_two # [chunk, window, channels=3, height=84, width=84]
                        
        item["task"] = torch.full(size=(idxs.shape[0], ), fill_value=TASK_DICT[self.task], dtype=torch.long) # [chunk]
        item["robot"] = torch.full(size=(idxs.shape[0], ), fill_value=ROBOT_DICT[self.robot], dtype=torch.long) # [chunk]
        item["idxs"] = idxs # [chunk, window]

        return item