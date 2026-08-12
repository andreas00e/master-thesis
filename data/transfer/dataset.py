import os 
import h5py 
from typing import Dict, List, Optional, Tuple

import torch 
from torchtyping import TensorType 
from torch.utils.data import Dataset 

from data.discover.utils.transforms import get_transforms


class TransferDataset(Dataset): 
    def __init__(self, 
        demo_map: List[Tuple[os.PathLike, int, int]], 
        horizon: int,
        crop_factor: float, 
        depth: Optional[bool], 
        joint_dsc: Dict[str, TensorType["*"]], 
        transforms: List[str] 
        ) -> None:
        super().__init__()
        
        self.demo_map = demo_map
        self.horizon = horizon
        self.crop_factor = crop_factor
        self.depth = depth
        self.joint_dsc = joint_dsc
        self.transforms = transforms
        
        self.transforms = get_transforms(self.transforms)
        
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
    
    def __getitem__(self, idx: int) -> Dict[str, TensorType["*"]]: 
        item = {}
        file, demo, _ = self.demo_map[idx] 
        
        robot = file.split(".")[0].split("_")[-2] if self.depth else file.split(".")[0].split("_")[-1]
        joint_dsc = self.joint_dsc[robot].T # [7, 3]
        
        if file not in self._file_cache: # open file once per worker and cache it
            self._file_cache[file] = h5py.File(file, "r")
            
        hf = self._file_cache[file]
    
        actions = hf["data"][demo]["actions"][()]
        actions = torch.from_numpy(actions).to(torch.float32) # [n, d]
        
        obs = hf["data"][demo]["obs"] 
        
        rgb_obs = obs["robot0_eye_in_hand_image"][()] # [n, h=84, w=84, c=3]
        rgb_obs = rgb_obs[:, :int(rgb_obs.shape[1]*self.crop_factor), ...] # [n, h=63, w=84, c=3]
        rgb_obs = torch.from_numpy(rgb_obs).permute(0, 3, 1, 2) # [n, c=3, h=84, w=84]
        rgb_obs = self.transforms(rgb_obs) # [n, c=3, h=224, w=224]
        
        joint_pos = obs["robot0_joint_pos"][()]
        joint_pos = torch.from_numpy(joint_pos).unsqueeze(-1) # [n, joints, 1]
        joint_vel = obs["robot0_joint_vel"][()]
        joint_vel = torch.from_numpy(joint_vel).unsqueeze(-1) # [n, joints, 1]
        joint_obs = torch.concat(tensors=(joint_pos, joint_vel), dim=-1).to(torch.float32) # [n, joints, 2]
       
        item["actions"] = actions
        item["rgb_obs"] = rgb_obs
        item["joint_dsc"] = joint_dsc
        item["joint_obs"] = joint_obs
        
        return item