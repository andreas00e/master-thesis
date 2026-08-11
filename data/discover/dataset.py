import os 
import h5py 
import numpy as np
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
        file, demo, _ =  self.demo_map[idx]
        
        robot = file.split(".")[0].split("_")[-1]
        task = file.split(".")[0].split("/")[-1].split("_")[0]

        hf = self._file_cache.get(file)
        if hf is None:
            hf = h5py.File(file, "r")
            self._file_cache[file] = hf

        rgb_obs = hf["data"][demo]["obs"]["robot0_eye_in_hand_image"]
        rgb_obs = rgb_obs[:, :int(rgb_obs.shape[1]*self.crop_factor), ...]
        rgb_obs = torch.from_numpy(rgb_obs).permute(0, 3, 1, 2) # [n, c, h, w]
                        
        rgb = self.transforms(rgb_obs) 
        rgb_plus = self.transforms(rgb_obs)
        
        # if self.depth: 
        #     depth_obs = hf["data"][demo]["obs"]["robot0_eye_in_hand_depth"]
        #     depth_obs = depth_obs[:, :int(depth_obs.shape[1]*self.crop_factor), ...]
        #     depth_obs = torch.from_numpy(depth_obs).permute(0, 3, 1, 2) # [n, 1, h, w]
            
        #     depth = self.transforms(depth_obs) 
        #     depth_plus = self.transforms(depth_obs)
        #     item["depth"] = depth[idxs]
        #     item["depth_plus"] = depth_plus[idxs]
            
        gripper_qpos = hf["data"][demo]["obs"]["robot0_gripper_qpos"][()] # [n, d]: d in {2, 6}
        gripper_min = self.df_gripper[f"{robot}_min"].values[:gripper_qpos.shape[1]][None, :] # [1, d]
        gripper_max = self.df_gripper[f"{robot}_max"].values[:gripper_qpos.shape[1]][None, :] # [1, d]
        
        gripper_qpos = np.clip((gripper_qpos - gripper_min) / (gripper_max - gripper_min), 0, 1) # [n, d]
        gripper_qpos = np.mean(gripper_qpos, axis=-1)
        gripper_qpos = torch.from_numpy(gripper_qpos, dtype=torch.float32)
    
        offsets = torch.randint(low=0, high=rgb_obs.shape[0]-self.window-1, size=(self.chunks, )) # indexing, not slicing
        idxs = offsets[:, None] + torch.arange(self.window) # [chunks, window]
        
        item["rgb"] = rgb[idxs, ...]
        item["rgb_plus"] = rgb_plus[idxs, ...] 
        item["gripper_qpos"] = gripper_qpos[idxs]
        item["idxs"] = idxs
        return item 
    
    
    # def _test_item(self, robot_one, robot_two, task_one, task_two): 
    #     idx = 0 
    #     item = {}
    #     file, demo, _ =  self.demo_map[0]
    #     assert self.state.stage == "test", "Wrong stage has been selected!"
        
    #     file_one, demo_one, _ = next((file, demo, _ for _ in self.demo_map if (file.split(".")[0].split("_")[-1] == robot_one and file.split(".")[0].split("/")[-1].split("_")[0] == task_one)), None)
    #     robot = file.split(".")[0].split("_")[-1]

        
    #     if not robot_two and not task_two: # same robot, same task 
    #         file_two = file_one
    #         idx_demo_one = int(demo_one.replace("demo_", ""))
    #         idx_demo_two = torch.randint(low=0, high=1000, size=(1, ))
            
    #     elif robot_two and not task_two: # different robots, same task 
    #         file_two = next((file, demo, _ for _ in self.demo_map if (file.split(".")[0].split("_")[-1] != robot_one and file.split(".")[0].split("/")[-1].split("_")[0] == task_one)), None)
    #         idx_demo_two = 5

    #     elif not robot_two and task_two: # same robot, different task 
    #         file_two = next((file, demo, _ for _ in self.demo_map if (file.split(".")[0].split("_")[-1] == robot_one and file.split(".")[0].split("/")[-1].split("_")[0] != task_one)), None)
    #         idx_demo_two = 5 
            
    #     elif robot_two and task_two: # different robot, different task 
    #         file_two = next((file, demo, _ for _ in self.demo_map if (file.split(".")[0].split("_")[-1] != robot_one and file.split(".")[0].split("/")[-1].split("_")[0] != task_one)), None)
    #         idx_demo_two = 5 

    #     hf_one = self._file_cache.get(file_one)
    #     if hf_one is None:
    #         hf_one = h5py.File(file_one, "r")
    #         self._file_cache[file_one] = hf_one

    #     hf_two = self._file_cache.get(file_one)
    #     if hf_two is None:
    #         hf_two = h5py.File(file_two, "r")
    #         self._file_cache[file_two] = hf_two
            
    #     rgb_obs_one = hf_one["data"][demo]["obs"]["robot0_eye_in_hand_image"]
    #     rgb_obs_one = rgb_obs_one[:, :int(rgb_obs_one.shape[1]*self.crop_factor), ...]
    #     rgb_obs_one = torch.from_numpy(rgb_obs_one).permute(0, 3, 1, 2) # [n, c, h, w]
                        
    #     rgb = self.transforms(rgb_obs_one) 
    #     rgb_plus = self.transforms(rgb_obs_one)
    #     gripper_qpos = hf_one["data"][demo]["obs"]["robot0_gripper_qpos"][()] # [n, d]: d in {2, 6}
    #     gripper_min = self.df_gripper[f"{robot}_min"].values[:gripper_qpos.shape[1]][None, :] # [1, d]
    #     gripper_max = self.df_gripper[f"{robot}_max"].values[:gripper_qpos.shape[1]][None, :] # [1, d]
        
    #     gripper_qpos = np.clip((gripper_qpos - gripper_min) / (gripper_max - gripper_min), 0, 1) # [n, d]
    #     gripper_qpos = np.mean(gripper_qpos, axis=-1)
    #     gripper_qpos = torch.from_numpy(gripper_qpos)
    
    #     offsets = torch.randint(low=0, high=rgb_obs_one.shape[0]-self.window-1, size=(self.chunks, )) # indexing, not slicing
    #     idxs = offsets[:, None] + torch.arange(self.window) # [chunks, window]
        