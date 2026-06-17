import os 
import h5py 
import numpy as np 
from typing import List, Optional

from torch.utils.data import Dataset 


class MimicGenRobotDataset(Dataset): 
    def __init__(self, 
        files: List[os.PathLike], 
        demo_map: List[List[os.PathLike, int, int, None]],
        horizon: Optional[int]=16,
        depth: Optional[bool]=False, 
        expand_depth: Optional[str]=None, # grayscale, colormap 
        *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        self.files = files
        self.demo_map = demo_map
        self.horizon = horizon
        self.depth = depth
        self.expand_detph = expand_depth 
        
        self.len, self.epoch = 0, 0
        self.demo_map, self.windows, self._rgb_views, self._depth_views = [], [], [], []        
        self._file_cache = {}
                       
    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
        rng = np.random.default_rng(epoch)

        for demo in self.demo_map:
            len_episode = demo[2]
            demo[3] = int(rng.integers(0, len_episode - self.horizon))
                           
    def __len__(self) -> int:
        return len(self.len)
    
    def __getstate__(self):
        state = self.__dict__.copy() # when the worker is forked/pickled, clear the cache 
        state["_file_cache"] = {}
        
        return state
    
    def __getitem__(self, idx): 
        out = {}  
        file, demo, len_episode, idx = self.demo_map[idx] 
              
        if file not in self._file_cache: # open file once per worker and cache it
            self._file_cache[file] = h5py.File(os.path.join(self.data_dir, file), "r")
            
        hf = self._file_cache[file]
        data = hf["data"][demo]
        obs = data["obs"]    

        rgb_obs = obs["robot0_eye_in_hand_image"][idx:idx+self.horizon, ...] # [horizon, H, W, 3]
        rgb_obs = rgb_obs.astype(np.float32) / 255.0
        out["rgb_obs"] = rgb_obs
    
        if self.depth: 
            d_obs = obs["robot0_eye_in_hand_depth"][idx:idx+self.horizon, ...] # [horizon, H, W, 1]
            out["d_obs"] = d_obs
            # TODO: add normalization of depth image
        
        out["file"] = "_".join(file.split(".")[0].split("_")[-1::-2])+"_"+str(idx)
        return out