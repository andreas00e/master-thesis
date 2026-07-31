import os 
import h5py 
import numpy as np 
from typing import Dict, List, Tuple

import torch 
from torchtyping import TensorType 
from torch.utils.data import Dataset 

class TransferDataset(Dataset): 
    def __init__(self, 
        demo_map: List[Tuple[os.PathLike, int, int]],
        window: int,
        joint_dsc: Dict[str, TensorType["*"]],
        ) -> None:
        super().__init__()
        
        self.demo_map = demo_map
        self.window = window
        self.joint_dsc = joint_dsc
        
        self.depth = True
        
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
    
    def __getitem__(self, idx) -> Dict[str, TensorType["*"]]: 
        item = {}
        file, demo, n_steps = self.demo_map[idx] 
        idx_start = np.random.randint(0, n_steps - self.window + 1)
        
        robot = file.split(".")[0].split("_")[-2] if self.depth else file.split(".")[0].split("_")[-1]
        joint_dsc = self.joint_dsc[robot]

        if file not in self._file_cache: # open file once per worker and cache it
            self._file_cache[file] = h5py.File(file, "r")
            
        hf = self._file_cache[file]
        obs = hf["data"][demo]["obs"]
        
        joint_pos = obs["robot0_joint_pos"][idx_start:idx_start+self.window, :] # [window, 3]
        joint_vel = obs["robot0_joint_vel"][idx_start:idx_start+self.window, :] # [window, 3]
        joint_obs = torch.vstack((joint_pos, joint_vel))
        
        item["joint_dsc"] = joint_dsc
        item["joint_obs"] = joint_obs 
        
        return item