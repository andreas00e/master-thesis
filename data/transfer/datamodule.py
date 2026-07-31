import os 
import h5py 
import numpy as np 
from omegaconf import OmegaConf
from typing import  Dict, List, Optional, Tuple

import torch
from torchtyping import TensorType
import lightning as pl 
from torch.utils.data import Dataset, DataLoader, random_split

from data.transfer.dataset import TransferDataset

class TransferDataModule(pl.LightningDataModule): 
    def __init__(self, 
        data_dir: os.PathLike, # directory containing the hdf5 trajectory files 
        window: int,
        robots: Optional[List[str]], 
        tasks: Optional[List[str]], 
        config_file_dir: os.PathLike, 
        batch_size: int,
        shuffle: bool,  
        num_workers: int, 
        pin_memory: bool, 
        persistent_workers: bool,
        dataset_lengths: List[int], 
        *args, **kwargs) -> None:
        super().__init__()
        
        # data kwargs
        self.data_dir = data_dir
        self.window = window
        self.robots = robots 
        self.tasks = tasks 
        self.config_file_dir = config_file_dir
        
        # dataloading kwargs
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.dataset_lengths = dataset_lengths

        all_files = [file for file in os.listdir(self.data_dir) if "depth" in file]
        all_robots = list(set(f.split(".")[-2].split("_")[-1] for f in all_files)) # no given robot -> all robots
        all_tasks = list(set(f.split(".")[-2].split("_")[0] for f in all_files)) # no given task -> all tasks
        
        self.robots = self._filtered_or_all(self.robots, all_robots)
        self.tasks  = self._filtered_or_all(self.tasks, all_tasks)
        self.files = [os.path.join(data_dir, file) for file in all_files
            if any(robot in file for robot in self.robots)
            and any(task in file for task in self.tasks)]
        
        self.demo_map = None
        self.joint_descr = None 
        self.train_dataset, self.val_dataset, self.test_dataset = self.setup()
        
        self.n_samples_per_epoch = len(self.demo_map)
    
    def _filtered_or_all(self, selected, available):
        if selected is None:
            return available
        filtered = [x for x in selected if x in available]
        return filtered if filtered else available
    
    def _get_demomap(self): # -> List[List[os.PathLike, int, int, None]]: 
        H = np.inf # H: min episode duration -> max possible window size
        demo_map = []

        for file in self.files:
            with h5py.File(file, "r") as hf:
                for demo in hf["data"].keys():
                    n_steps = hf["data"][demo]["actions"][()].shape[0]
                    if n_steps < H:
                        H = n_steps
                    demo_map.append([file, demo, n_steps])
        
        if H < self.window: 
            print(f"The chosen size of the window is bigger than the smallest episode length! \n \
                  Therefore, the size of the window gets changed from {self.window} to {H}.")
            self.window = H
        
        return demo_map
    
    def _get_joint_descr(self) -> Dict[str: TensorType["*"]]: 
        files = [os.path.join(self.config_file_dir, file) for file in os.listdir(self.config_file_dir)]
        cfgs = [OmegaConf.to_container(OmegaConf.load(file), resolve=True) for file in files]
        
        joint_descr = {}
        for cfg in cfgs: 
            robot = next(iter(cfg))
            cfg = cfg[robot]
            joint_descr[self.robots] = torch.stack([torch.tensor(value) for value in cfg.values()], dim=0)
            
        return joint_descr
        
    def setup(self, stage=None) -> Tuple[Dataset, Dataset, Dataset]:
        self.demo_map = self._get_demomap()
        self.joint_descr = self._get_joint_descr()
        
        dataset = TransferDataset(
            demo_map=self.demo_map,
            window=self.window,
            config_path_dh=self.config_path_dh, 
            )
        
        train_dataset, val_dataset, test_dataset = random_split(dataset, lengths=self.dataset_lengths)
        return train_dataset, val_dataset, test_dataset
    
    def train_dataloader(self):
        train_dataloader = DataLoader(
            dataset=self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=self.shuffle,    
            num_workers=self.num_workers,  
            pin_memory=self.pin_memory, 
            persistent_workers=self.persistent_workers, 
            )
        return train_dataloader
    
    def val_dataloader(self):
        val_dataloader = DataLoader(
            dataset=self.val_dataset, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers, 
            pin_memory=self.pin_memory, 
            persistent_workers=self.persistent_workers, 
            )
        return val_dataloader
    
    def test_dataloader(self):
        test_dataloader = DataLoader(
            dataset=self.val_dataset, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers, 
            pin_memory=self.pin_memory, 
            persistent_workers=self.persistent_workers, 
            )
        return test_dataloader