import os 
import h5py 
import pandas as pd
import numpy as np 
from typing import Dict, List, Tuple

import torch
from torchtyping import TensorType 
from torch.utils.data import Dataset 
from torchvision.tv_tensors import Video 

from data.discover.utils.transforms import get_transforms


class MimicGenRobotDataset(Dataset): 
    def __init__(self, 
        demo_map: List[Tuple[os.PathLike, int, int]],
        depths: pd.DataFrame,
        window: int,
        expand_depth: str, # grayscale, colormap 
        transforms: List[str]
        ) -> None:
        super().__init__()
        
        self.demo_map = demo_map
        self.depths = depths
        self.window = window
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
    
    def __del__(self):
      for f in self._file_cache.values():
          try:
              f.close()
          except Exception:
              pass
    
    def __getitem__(self, idx):
        rgb_obs_lst = []
        rgb_obs_plus_lst = []

        for file, demo, n_steps in self.demo_map[idx]:
            start = np.random.randint(0, n_steps - self.window + 1)

            hf = self._file_cache.get(file)
            if hf is None:
                hf = h5py.File(file, "r")
                self._file_cache[file] = hf

            obs = hf["data"][demo]["obs"]["robot0_eye_in_hand_image"][start:start + self.window] # [window, height=84, width=84, channels=3]
            rgb = torch.from_numpy(obs).permute(0, 3, 1, 2).contiguous()  
            rgb_plus = self.transforms_plus(rgb)
            rgb = self.transforms(rgb)

            rgb_obs_lst.append(rgb)
            rgb_obs_plus_lst.append(rgb_plus)

        return torch.stack(rgb_obs_lst), torch.stack(rgb_obs_plus_lst)