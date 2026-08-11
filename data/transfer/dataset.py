import os 
import h5py 
from typing import Dict, List, Optional, Tuple

import torch 
from torchtyping import TensorType 
from torch.utils.data import Dataset 


class TransferDataset(Dataset): 
    def __init__(self, 
        demo_map: List[Tuple[os.PathLike, int, int]], 
        horizon: int,
        depth: Optional[bool], 
        joint_dsc: Dict[str, TensorType["*"]] 
        ) -> None:
        super().__init__()
        
        self.demo_map = demo_map
        self.horizon = horizon
        self.depth = depth
        self.joint_dsc = joint_dsc
        
        self._file_cache = {}
                       
    def __len__(self) -> int:
        return len(self.demo_map)
    
    def __getstate__(self) -> Dict:
        state = self.__dict__.copy() # when the worker is forked/pickled, clear the cache 
        state["_file_cache"] = {}
        return state
    
    def __del__(self):
      for f in self._file_cache.values():
          try:
              f.close()
          except Exception:
              pass
    
    def __getitem__(self, idx: int) -> Dict[str, TensorType["*"]]: 
        item = {}
        file, demo, _ = self.demo_map[idx] 
        
        robot = file.split(".")[0].split("_")[-2] if self.depth else file.split(".")[0].split("_")[-1]
        joint_dsc = self.joint_dsc[robot] # [features, joints]

        if file not in self._file_cache: # open file once per worker and cache it
            self._file_cache[file] = h5py.File(file, "r")
            
        hf = self._file_cache[file]
    
        actions = hf["data"][demo]["actions"][()]
        actions = torch.from_numpy(actions)
        
        obs = hf["data"][demo]["obs"] 
        
        rgb_obs = obs["robot0_eye_in_hand_image"][()]
        rgb_obs = torch.from_numpy(rgb_obs)
        
        joint_pos = obs["robot0_joint_pos"][()]
        joint_pos = torch.from_numpy(joint_pos)
        joint_vel = obs["robot0_joint_qpos"][()]
        joint_vel = torch.from_numpy(joint_vel)
        joint_obs = torch.vstack((joint_pos, joint_vel))
       
        item["actions"] = actions
        item["rgb_obs"] = rgb_obs
        item["joint_dsc"] = joint_dsc
        item["joint_obs"] = joint_obs
        
        return item