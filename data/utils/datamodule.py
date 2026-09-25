import os 
import random 
import pandas as pd
from pathlib import Path
from typing import List, Optional, Union

import torch
from torch.utils.data import DataLoader, ConcatDataset, random_split

import lightning.pytorch as pl 

from data.utils.collate import collate_discover
from data.utils.load_files import get_files, get_metadata, get_demo_dict
from data.utils.dataset import MimicGenRobotDataset
from data.discover.utils.sampler import SameRobotBatchSampler


class MimicGenRobotDataModule(pl.LightningDataModule): 
    def __init__(self, 
        data_dir: str, # directory containing the hdf5 trajectory files 
        meta_dir: str, # directory containing the hdf5 files metadata (e.g. min & max of depth maps)
        transforms_list: List[str], 
        contrastive_transforms: bool=False, 
        robots: Optional[Union[str, List[str]]]=None, 
        tasks: Optional[Union[str, List[str]]]=None, 
        data_distribution: float="d0", # d0 or d0 and d1
        data_portion: float=1.0, # percentage of dataset use
        window_size: int=8, 
        chunk_size: int=1, 
        crop_factor: float=1.0,       
        noise_level: float=1.0, 
        consistent_batch: bool=False, 
        temporal_smoothing: bool=False, 
        positive_window_size: Optional[int]=None,
        negative_window_size: Optional[int]=None,  
        batch_size: int=16,
        shuffle: bool=True,  
        num_workers: int=0, 
        pin_memory: bool=False, 
        persistent_workers: bool=True,
        drop_last: bool=False, 
        dataset_lengths: List[float]=[0.8, 0.1, 0.1],
        seed: int=42, 
        ) -> None:
        super().__init__()
       
        # Data kwargs
        self.data_dir = Path(data_dir)
        self.meta_dir = Path(meta_dir)
        
        # Image transformations/ augmenations
        self.transforms_list = transforms_list
        self.contrastive_transforms = contrastive_transforms
        
        self.data_distribution = data_distribution
        self.data_portion = data_portion
        self.window_size = window_size
        self.chunk_size = chunk_size
        self.crop_factor = crop_factor
        self.noise_level = noise_level
        
        self.consistent_batch = consistent_batch
        self.temporal_smoothing = temporal_smoothing
        self.positive_window_size = positive_window_size 
        self.negative_window_size = negative_window_size 

        # Dataloading kwargs
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers if num_workers < os.cpu_count() else max(1, os.cpu_count())
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.drop_last = drop_last
        self.dataset_lengths = dataset_lengths
        self.seed = seed
        
        # File handling 
        self.robots, self.tasks, self.files = get_files(self.data_dir, robots, tasks) # all hdf5 files containg given robot(s) and task(s)
        self.metadata = get_metadata(self.meta_dir, self.files)
        self.dataframe_gripper = pd.read_csv(self.meta_dir / "gripper_state_robot.csv")
        self.demo_map, self.window_size = get_demo_dict(self.metadata, self.files, self.window_size) # Tuple[Dict[str, List[Tuple[str, str, int]]], int]
            
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
       
    def setup(self, stage: Optional[str]=None) -> None:
        if getattr(self, "val_dataset", None) is not None: 
            self.teardown(stage=stage)
        
        rng = random.Random(self.seed)
           
        demo_map = {} 
        for robot in self.robots: 
            for task in self.tasks: 
                d0_key = f"{task}_d0_{robot}"
                d1_key = f"{task}_d1_{robot}"
                map_key = f"{task}{robot}"
                
                if self.data_distribution == "d0": 
                    entries = list(self.demo_map[d0_key])
                elif self.data_distribution == "d1": 
                    entries = list(self.demo_map[d1_key])
                elif self.data_distribution == "both":
                    entries = list(self.demo_map[d0_key] + self.demo_map[d1_key])
                else: 
                    raise ValueError(f"n_ds must 1 or 2, got {self.data_distribution}")
            
                if self.data_portion < 1: 
                    n_dm = len(entries)
                    rng.shuffle(entries)
                    entries = entries[:int(self.data_portion * n_dm)]
                
                demo_map[map_key] = entries
        
        datasets = [
            MimicGenRobotDataset(
            demo_map=demo_map[f"{task}{robot}"],
            dataframe_gripper=self.dataframe_gripper, 
            transforms_list=self.transforms_list, 
            contrastive_transforms=self.contrastive_transforms,
            window_size=self.window_size,
            chunk_size=self.chunk_size, 
            temporal_smoothing=self.temporal_smoothing, 
            positive_window_size=self.positive_window_size, 
            negative_window_size=self.negative_window_size, 
            crop_factor=self.crop_factor,
            noise_level=self.noise_level
            ) 
            for task in self.tasks 
            for robot in self.robots
        ]
        
        train_dataset = []
        val_dataset = []
        test_dataset = []
        
        generator = torch.Generator().manual_seed(self.seed)

        for dataset in datasets: 
            train_subset, val_subset, test_subset = random_split(dataset, lengths=self.dataset_lengths, generator=generator)
            train_dataset.append(train_subset)
            val_dataset.append(val_subset)
            test_dataset.append(test_subset)
        
        self.train_dataset = ConcatDataset(train_dataset)
        self.val_dataset = ConcatDataset(val_dataset)
        self.test_dataset = ConcatDataset(test_dataset)
            
    def teardown(self, stage: Optional[str]=None) -> None:
        for stage_ in ["train", "val", "test"]: 
            dataset = getattr(self,  f"{stage_}_dataset")
            
            if isinstance(dataset, ConcatDataset):
                for subset in dataset.datasets: 
                    base_set = getattr(subset, "dataset", subset)
                    
                    if hasattr(base_set, "close") and callable(base_set.close): 
                        base_set.close()
                                    
        self.train_dataset = None 
        self.val_dataset = None 
        self.test_dataset = None 
        
    def __del__(self):
        try:
            self.teardown()
        except Exception:
            pass
        
    def _make_dataloader(self, dataset, shuffle: bool) -> DataLoader:        
        batch_sampler = None 
         
        if self.consistent_batch: 
            batch_sampler = SameRobotBatchSampler(
                concat_dataset=dataset, 
                n_robots=len(self.robots),
                batch_size=self.batch_size, 
                shuffle=shuffle, 
                drop_last=self.drop_last
            )
            
            return DataLoader(
                dataset=dataset, 
                batch_sampler=batch_sampler, 
                num_workers=self.num_workers,  
                pin_memory=self.pin_memory, 
                persistent_workers=self.persistent_workers, 
                collate_fn=collate_discover,
                )
            
        return DataLoader(
            dataset=dataset, 
            batch_size=self.batch_size,
            shuffle=shuffle,
            batch_sampler=batch_sampler, 
            num_workers=self.num_workers,  
            pin_memory=self.pin_memory, 
            persistent_workers=self.persistent_workers, 
            drop_last=self.drop_last,
            collate_fn=collate_discover, 
            )
    
    def train_dataloader(self) -> DataLoader:
        return self._make_dataloader(self.train_dataset, shuffle=self.shuffle)
    
    def val_dataloader(self) -> DataLoader:
        return self._make_dataloader(self.val_dataset, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._make_dataloader(self.test_dataset, shuffle=False)