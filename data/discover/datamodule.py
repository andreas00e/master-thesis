import os 
import pandas as pd
from typing import List, Optional, Union

import torch
from torch.utils.data import DataLoader, ConcatDataset, random_split
import lightning.pytorch as pl 

from data.utils.load_files import get_files, get_metadata, get_demo_list
from data.discover.dataset import MimicGenRobotDataset
from data.utils.collate import collate_discover


class MimicGenRobotDataModule(pl.LightningDataModule): 
    def __init__(self, 
        data_dir: Union[str, os.PathLike], # directory containing the hdf5 trajectory files 
        meta_dir: Union[str, os.PathLike], # directory containing the hdf5 files metadata (e.g. min & max of depth maps)
        robots: Optional[Union[str, List[str]]], 
        tasks: Optional[Union[str, List[str]]], 
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
        dataset_lengths: List[int], 
        transforms: List[str], 
        ) -> None:
        super().__init__()
        
        if not 0 < noise_level < 1: raise ValueError(f"noise_level has to be in (0,1), got {noise_level}.")
        if num_workers > os.cpu_count(): raise ValueError(f"num_workers is bigger than actual limit, got {num_workers}.")
        if sum(dataset_lengths) != 1: raise ValueError(f"Sum of dataset lengths do not sum up to 1, got {sum(dataset_lengths)}.")
        if window < 1: raise ValueError(f"Window size has to be bigger than 1, got {window}.")
        if chunk < 1: raise ValueError(f"Chunk size has to be bigger than 1, got {chunk}.")
        
        # Data kwargs
        self.data_dir = data_dir
        self.meta_dir = meta_dir
        self.robots = list(robots) if isinstance(robots, str) else robots
        self.tasks = list(tasks) if isinstance(tasks, str) else tasks
        self.depth = depth
        self.crop_factor = crop_factor
        self.noise_level = noise_level
        self.window = window
        self.chunk = chunk

        # Dataloading kwargs
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        # self.multiprocessing_context = multiprocessing_context
        self.persistent_workers = persistent_workers
        self.dataset_lengths = dataset_lengths
        
        # Image augmenation pipeline
        self.transforms = transforms

        # File handling 
        self.files = get_files(self.data_dir, self.depth, self.robots, self.tasks) # all hdf5 files containg given robot(s) and task(s)
        self.meta_data = get_metadata(self.meta_dir, self.files)
        self.df_gripper = pd.read_csv(os.path.join(self.meta_dir, "gripper_state_robot.csv"))
        self.demo_map, self.window = get_demo_list(self.meta_data, self.files, self.window)
            
        self.dataset_ = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
       
    def setup(self, stage: Optional[str]=None) -> None:
        if getattr(self, "dataset_", None) is not None: 
            self.teardown(stage=stage)
            
        datasets = [
            MimicGenRobotDataset(
            demo_map=self.demo_map[f"{str(task)}_{str(robot)}"],
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
        
        generator = torch.Generator().manual_seed(getattr(self, "seed", 42))
                
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
        except:
            pass  
        
    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=self.shuffle,    
            num_workers=self.num_workers,  
            pin_memory=self.pin_memory, 
            # multiprocessing_context=self.multiprocessing_context, 
            persistent_workers=self.persistent_workers, 
            collate_fn=collate_discover  
            )
    
    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers, 
            pin_memory=self.pin_memory, 
            # multiprocessing_context=self.multiprocessing_context, 
            persistent_workers=self.persistent_workers, 
            collate_fn=collate_discover 
            )
    
    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers, 
            pin_memory=self.pin_memory, 
            # multiprocessing_context=self.multiprocessing_context, 
            persistent_workers=self.persistent_workers, 
            collate_fn=collate_discover,    
            )
        
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"robots={self.robots}, "
            f"tasks={self.tasks}, "
            f"depth={self.depth}, "
            f"batch_size={self.batch_size}, "
            f"dataset_lengths={self.dataset_lengths})"
        )