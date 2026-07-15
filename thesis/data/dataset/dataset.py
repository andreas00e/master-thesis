import os 
import h5py 
import numpy as np 
from typing import List, Optional

from torch.utils.data import Dataset 

from data.utils import get_transforms


class MimicGenRobotDataset(Dataset): 
    def __init__(self, 
        files, # : List[os.PathLike], # all hdf5 files
        demo_map, #: List[os.PathLike, int, int],
        window: int,
        expand_depth: Optional[str] = None, # grayscale, colormap 
        *args, **kwargs) -> None:
        super().__init__()
        
        self.files = files
        self.demo_map = demo_map
        self.window = window
        self.expand_detph = expand_depth 
        
        self.len, self.epoch = 0, 0
        self.windows, self.rgb_views, self.depth_vies = [], [], [] 
        self.file_chache = {}
                       
    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
        rng = np.random.default_rng(epoch)

        for demo in self.demo_map:
            len_episode = demo[2]
            demo[3] = int(rng.integers(0, len_episode - self.window))
                           
    def __len__(self) -> int:
        return len(self.demo_map)
    
    def __getstate__(self):
        state = self.__dict__.copy() # when the worker is forked/pickled, clear the cache 
        state["_file_cache"] = {}
        return state
    
    def __getitem__(self, idx): 
        item = {}  
        file, demo, n_steps = self.demo_map[idx] 
        idx_start = np.random.randint(0, n_steps-self.window)
        
        if file not in self.file_chache: # open file once per worker and cache it
            self.file_chache[file] = h5py.File(file, "r")
            
        hf = self.file_chache[file]
        data = hf["data"][demo]
        obs = data["obs"] 
        
        rgb_obs = obs["robot0_eye_in_hand_image"][idx_start:idx_start+self.window, ...]
        item["rgb_obs"] = rgb_obs.astype(np.float32) / 255.0
        
        depth_obs = obs["robot0_eye_in_hand_depth"][idx_start:idx_start+self.window, ...]
        item["depth_obs"] = depth_obs 
                              
        item["file"] = "_".join(file.split(".")[0].split("_")[-1::-2])+"_"+str(idx)
        return item