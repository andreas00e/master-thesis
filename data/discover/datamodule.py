import os 
import math
import random 
import pandas as pd
from pathlib import Path
from omegaconf import ListConfig
from typing import List, Optional, Union

import torch
from torch.utils.data import DataLoader, ConcatDataset, random_split
import lightning.pytorch as pl 

from data.utils.load_files import get_files, get_metadata, get_demo_dict
from data.discover.dataset import MimicGenRobotDataset
from data.utils.collate import collate_discover


class MimicGenRobotDataModule(pl.LightningDataModule): 
    def __init__(self, 
        data_dir: Union[str, os.PathLike], # directory containing the hdf5 trajectory files 
        meta_dir: Union[str, os.PathLike], # directory containing the hdf5 files metadata (e.g. min & max of depth maps)
        robots: Optional[Union[str, List[str]]], 
        tasks: Optional[Union[str, List[str]]], 
        n_ds: int, # d0 or d0 and d1
        f_ds: float, 
        depth: bool, 
        crop_factor: float,        
        noise_level: float, 
        window: int, 
        chunk: int, 
        batch_size: int,
        shuffle: bool,  
        num_workers: int, 
        pin_memory: bool, 
        persistent_workers: bool,
        dataset_lengths: List[float], 
        seed: int,
        transforms: List[str], 
        ) -> None:
        super().__init__()
        
        if not n_ds in [1, 2]: 
            raise ValueError(f"n_ds has to be 1 or 2, got {n_ds}.")
        if not 0 < f_ds <= 1: 
            raise ValueError(f"f_ds has to be in (0, 1], got {f_ds}.")
        if not 0 < crop_factor < 1: 
            raise ValueError(f"crop_factor has to be in (0, 1), got {crop_factor}.")
        if not 0 < noise_level < 1: 
            raise ValueError(f"noise_level has to be in (0, 1), got {noise_level}.")
        if window < 1: 
            raise ValueError(f"Size of window must be >= 1, got {window}.")
        if chunk < 1: 
            raise ValueError(f"Chunk size must be >=1,  got {chunk}.")
        
        cpu_count = os.cpu_count() or 1
        if num_workers > cpu_count: 
            self.num_workers = cpu_count
        else: 
            self.num_workers = num_workers
        if not math.isclose(sum(dataset_lengths), 1.0, rel_tol=1e-5):
            raise ValueError(f"Sum of dataset lengths must be 1.0, got {sum(dataset_lengths)}.")
       
        # Data kwargs
        self.data_dir = Path(data_dir)
        self.meta_dir = Path(meta_dir)
        
        self.n_ds = n_ds
        self.f_ds = f_ds
        self.depth = depth
        self.crop_factor = crop_factor
        self.noise_level = noise_level
        self.window = window
        self.chunk = chunk

        # Dataloading kwargs
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.dataset_lengths = dataset_lengths
        self.seed = seed

        # Image transformations/ augmenations
        self.transforms = transforms

        # File handling 
        self.robots, self.tasks, self.files = get_files(self.data_dir, self.depth, robots, tasks) # all hdf5 files containg given robot(s) and task(s)
        self.metadata = get_metadata(self.meta_dir, self.files)
        self.df_gripper = pd.read_csv(self.meta_dir / "gripper_state_robot.csv") # TODO: Change!
        self.demo_map, self.window = get_demo_dict(self.metadata, self.files, self.window)  # Tuple[Dict[str, List[Tuple[str, str, int]]], int]
            
        self.dataset_ = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
       
    def setup(self, stage: Optional[str]=None) -> None:
        if getattr(self, "dataset_", None) is not None: 
            self.teardown(stage=stage)
        
        rng = random.Random(self.seed)
           
        demo_map = {} 
        for robot in self.robots: 
            for task in self.tasks: 
                d0_key = f"{task}_d0_{robot}"
                d1_key = f"{task}_d1_{robot}"
                map_key = f"{task}{robot}"
                
                if self.n_ds == 1: 
                    entries = list(self.demo_map[d0_key])
                elif self.n_ds == 2: 
                    entries = list(self.demo_map[d0_key] + self.demo_map[d1_key])
                else: 
                    raise ValueError(f"n_ds must 1 or 2, got {self.n_ds}")
            
                if self.f_ds < 1: 
                    n_dm = len(entries)
                    rng.shuffle(entries)
                    entries = entries[:int(self.f_ds * n_dm)]
                
                demo_map[map_key] = entries
        
        datasets = [
            MimicGenRobotDataset(
            demo_map=demo_map[f"{task}{robot}"],
            df_gripper=self.df_gripper, 
            window=self.window,
            chunk=self.chunk, 
            crop_factor=self.crop_factor,
            noise_level=self.noise_level,
            transforms=self.transforms
            ) 
            for task in self.tasks 
            for robot in self.robots
        ]
        
        if not datasets: 
            raise RuntimeError("No datasets were constructed")
        
        generator = torch.Generator().manual_seed(self.seed)
                
        self.dataset_ = ConcatDataset(datasets)
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            self.dataset_, lengths=self.dataset_lengths, generator=generator
            )
    
    def teardown(self, stage: Optional[str]=None) -> None:
        dataset_ = getattr(self, "dataset_", None)
        
        if dataset_ is not None: 
            datasets_ = getattr(dataset_, "datasets", [dataset_]) 

            for ds in datasets_:
                if (hasattr(ds, "close") and callable(ds.close)): 
                    ds.close()
                                    
        self.dataset_ = None 
        self.train_dataset = None 
        self.val_dataset = None 
        self.test_dataset = None 
        
    def __del__(self):
        try:
            self.teardown()
        except Exception:
            pass
        
    def _make_dataloader(self, dataset, shuffle: bool=False) -> DataLoader: 
        return DataLoader(
            dataset=dataset, 
            batch_size=self.batch_size, 
            shuffle=shuffle,    
            num_workers=self.num_workers,  
            pin_memory=self.pin_memory, 
            persistent_workers=self.persistent_workers if self.num_workers > 0 else False, 
            collate_fn=collate_discover, 
            multiprocessing_context="fork"  
            )
    
    def train_dataloader(self) -> DataLoader:
        return self._make_dataloader(self.train_dataset, shuffle=self.shuffle)
    
    def val_dataloader(self) -> DataLoader:
        return self._make_dataloader(self.val_dataset, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._make_dataloader(self.test_dataset, shuffle=False)
        
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"robots={self.robots},"
            f"tasks={self.tasks},"
            f"depth={self.depth},"
            f"batch_size={self.batch_size},"
            f"dataset_lengths={self.dataset_lengths})"
        )