import os 
import h5py 
import numpy as np
import pandas as pd 
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
from torchtyping import TensorType
from torch.utils.data import Dataset

from data.discover.utils.transforms import get_transforms


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


class TransferDataset(Dataset):
    def __init__(
        self,
        demo_map: List[Tuple[Union[str, Path], str, int]],
        joint_dsc: Dict[str, TensorType["*"]], 
        dataframe_gripper: pd.DataFrame, 
        transforms_list: List[str], 
        action_horizon: int, 
        condition_horizon: int, 
        crop_factor: float=1.0, 
        )-> None: 
        super().__init__()
        
        self.demo_map = demo_map
        self.joint_dsc = joint_dsc
        self.dataframe_gripper = dataframe_gripper
        self.transforms_list = transforms_list
        self.action_horizon = action_horizon
        self.condition_horizon = condition_horizon
        self.crop_factor = crop_factor

        self._file_cache: Dict[str, h5py.File] = {}  
        self._pid: Optional[int] = None

        self.transforms = get_transforms(self.transforms_list)  

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
        item = {}
        file_path, demo, _ =  self.demo_map[idxs] # one hdf5 file 
        
        task = Path(file_path).stem.split("_")[0]
        robot = Path(file_path).stem.split("_")[-1]
        
        hf = self._get_hdf5_handle(file_path)
        demo_obs = hf["data"][demo]["obs"]
        
        n_steps = demo_obs["robot0_eye_in_hand_image"].shape[0]
        
        actions_max_start = max(1, n_steps - self.action_horizon + 1)
        actions_start = np.random.randint(0, actions_max_start) 
        actions_idxs = actions_start + np.arange(self.action_horizon) # [action_horizon]
        actions_idxs = np.clip(actions_idxs, 0, n_steps-1)
        action_values, action_counts = np.unique(actions_idxs, return_counts=True)

        actions = hf["data"][demo]["actions"][action_values] # [action_values, action_dim]
        actions = np.repeat(actions, action_counts, axis=0) # [action_horizon, action_dim]
        actions = torch.from_numpy(actions).to(torch.float32)

        conditions_idxs = np.arange(actions_start-self.condition_horizon, actions_start) # [condition_horizon]
        conditions_idxs = np.clip(conditions_idxs, 0, None)
        condition_values, condition_counts = np.unique(conditions_idxs, return_counts=True)

        # 1. Perspective 1: robot_0_eye_in_hand_image
        rgb_one = demo_obs["robot0_eye_in_hand_image"][condition_values] # [condition_values, height=84, width=84, channels=3]
        rgb_one = np.repeat(rgb_one, condition_counts, axis=0) # [condition_horizon, height=84, width=84, channels=3]
        if self.crop_factor is not None: 
            crop_h = int(rgb_one.shape[1]*self.crop_factor)
            rgb_one = rgb_one[:, :crop_h, ...]
        rgb_one = torch.from_numpy(rgb_one).permute(0, 3, 1, 2) # [condition_horizon, channels=3, height=224, width=224] 
        rgb_one = self.transforms(rgb_one)
        
        # 2. Perspective 2: agentview_image 
        rgb_two = demo_obs["agentview_image"][condition_values] # [condition_values, height=84, width=84, channels=3]   
        rgb_two = np.repeat(rgb_two, condition_counts, axis=0) # [condition_horizon, height=84, width=84, channels=3]    
        rgb_two = torch.from_numpy(rgb_two).permute(0, 3, 1, 2) # [condition_horizon, channels=3, height=224, width=224]    
        rgb_two = self.transforms(rgb_two)
        
        joint_dsc = self.joint_dsc[robot].T # [joint=7, 3]
        
        joint_pos = demo_obs["robot0_joint_pos"][condition_values] # [condition_values, joints]
        joint_pos = np.repeat(joint_pos, condition_counts, axis=0) # [condition_horizon, joints]
        joint_pos = torch.from_numpy(joint_pos) 
        joint_vel = demo_obs["robot0_joint_vel"][condition_values] # [condition_values, joints]
        joint_vel = np.repeat(joint_vel, condition_counts, axis=0) # [condition_horizon, joints]
        joint_vel = torch.from_numpy(joint_vel) # [condition_horizon, joints]
        joint_obs = torch.stack(tensors=(joint_pos, joint_vel), dim=-1).to(torch.float32) # [condition_horizon, joints, 2]   
        
        # 3. Normalized gripper joint states 
        g_qpos = demo_obs["robot0_gripper_qpos"][condition_values] # [condition_values, d]: d in {2, 6}
        g_qpos = np.repeat(g_qpos, condition_counts, axis=0)  # [condition_horizon, d]:
        min_col, max_col = f"{robot}_min", f"{robot}_max"
        
        if min_col in self.dataframe_gripper.columns and max_col in self.dataframe_gripper.columns: 
            g_min = self.dataframe_gripper[min_col].values[:g_qpos.shape[-1]] # [d]
            g_max = self.dataframe_gripper[max_col].values[:g_qpos.shape[-1]] # [d]
            g_qpos = np.clip((g_qpos - g_min) / ((g_max - g_min) + 1e-8), 0.0, 1.0) # [condition_horizon, d] 
            
        g_qpos = np.mean(g_qpos, axis=-1) # [condition_horizon]
        g_qpos = torch.from_numpy(g_qpos).to(torch.float32) .unsqueeze(-1)

        item["actions"] = actions # [action_horizon, action_dim]
        item["rgb_one"] = rgb_one # [condition_horizon, channels=3, height=224, width=224] 
        item["rgb_two"] = rgb_two # [condition_horizon, channels=3, height=224, width=224] 
        item["joint_dsc"] = joint_dsc # [joints=7, 3]
        item["joint_obs"] = joint_obs  # [condition_horizon, joints, 2]
        item["g_qpos"] = g_qpos # [condition_horizon]
        
        item["task"] = torch.full(size=(1, ), fill_value=TASK_DICT[task], dtype=torch.long) # []
        item["robot"] = torch.full(size=(1, ), fill_value=ROBOT_DICT[robot], dtype=torch.long) # []
        item["actions_idxs"] = torch.from_numpy(actions_idxs) # [action_horizon]
        item["conditions_idxs"] = torch.from_numpy(conditions_idxs) # [condition_horizon]

        return item