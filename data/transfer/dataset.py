import os 
import h5py 
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch 
from torchtyping import TensorType 
from torch.utils.data import Dataset 

from data.discover.utils.transforms import get_transforms


class TransferDataset(Dataset): 
    def __init__(self, 
        demo_map: List[Tuple[os.PathLike, int, int]], 
        crop_factor: float, 
        joint_dsc: Dict[str, TensorType["*"]], 
        transforms_list: List[str] 
        ) -> None:
        super().__init__()
        
        self.demo_map = demo_map
        self.crop_factor = crop_factor
        self.joint_dsc = joint_dsc
        self.transforms_list = transforms_list
        
        self.transforms = get_transforms(self.transforms_list)
        
        self._file_cache: Dict[str, h5py.File] = {}  
        self._pid: Optional[int] = None
        
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
        
        robot = Path(file_path).stem.split("_")[-1]
        
        hf = self._get_hdf5_handle(file_path)
        demo_obs = hf["data"][demo]["obs"]
        
        actions = hf["data"][demo]["actions"][()] # [seq_len, d]
        actions = torch.from_numpy(actions).to(torch.float32)
                
        rgb_one = demo_obs["robot0_eye_in_hand_image"][()] 
        rgb_one = rgb_one[:, :int(rgb_one.shape[1]*self.crop_factor), ...]
        rgb_one = torch.from_numpy(rgb_one).permute(0, 3, 1, 2)
        rgb_one = self.transforms(rgb_one)
        
        rgb_two = demo_obs["agentview_image"][()] 
        rgb_two = rgb_two[:, :int(rgb_two.shape[1]*self.crop_factor), ...]
        rgb_two = torch.from_numpy(rgb_two).permute(0, 3, 1, 2)
        rgb_two = self.transforms(rgb_two)
        
        joint_dsc = self.joint_dsc[robot].T # [7, 3]
        
        joint_pos = demo_obs["robot0_joint_pos"][()]
        joint_pos = torch.from_numpy(joint_pos).unsqueeze(-1) # [seq_len, joints, 1]
        joint_vel = demo_obs["robot0_joint_vel"][()]
        joint_vel = torch.from_numpy(joint_vel).unsqueeze(-1) # [seq_len, joints, 1]
        joint_obs = torch.concat(tensors=(joint_pos, joint_vel), dim=-1).to(torch.float32) # [seq_len, joints, 2]

        item["actions"] = actions
        item["rgb_one"] = rgb_one
        item["rgb_two"] = rgb_two
        item["joint_dsc"] = joint_dsc
        item["joint_obs"] = joint_obs
        
        return item