import os 
import h5py 
import numpy as np 
from typing import Dict, List, Tuple

from torchtyping import TensorType 
from torch.utils.data import Dataset 

class TransferDataset(Dataset): 
    def __init__(self, 
        demo_map: List[Tuple[os.PathLike, int, int]],
        window: int,
        joint_descr: Dict[str, TensorType["*"]],
        ) -> None:
        super().__init__()
        
        self.demo_map = demo_map
        self.window = window
        self.joint_descr = joint_descr
        
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
    
    def __getitem__(self, idx) -> Tuple[TensorType["*"], ...]: 
        file, demo, n_steps = self.demo_map[idx] 
        idx_start = np.random.randint(0, n_steps - self.window + 1)
        
        
        
        
        if file not in self._file_cache: # open file once per worker and cache it
            self._file_cache[file] = h5py.File(file, "r")
            
        hf = self._file_cache[file]
        data = hf["data"][demo]
        obs = data["obs"] 
        
                
        rgb_obs = obs["robot0_eye_in_hand_image"][idx_start:idx_start+self.window, ...] # [window, height=84, width=84, channels=3]
        rgb_obs = np.transpose(rgb_obs, (0, 3, 1, 2)) 
        
        return rgb_obs