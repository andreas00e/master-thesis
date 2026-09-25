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
        data_dir: str, 
        meta_dir: str,
        transforms_list: List[str], 
        contrastive_transforms: bool=False, 
        robots: Optional[Union[str, List[str]]]=None, 
        tasks: Optional[Union[str, List[str]]]=None, 
        data_distribution: str="d0", 
        data_portion: float=1.0,
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
        rng = random.Random(self.seed)
           
        demo_map = {"fit": {}, "validate": {}, "test": {}}         
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
                
                rng.shuffle(entries)
                n_entries = len(entries)
                if self.data_portion < 1: 
                    entries = entries[:int(self.data_portion * n_entries)]
                
                n_entries = len(entries)
                fit_end = int(self.dataset_lengths[0]*n_entries)
                val_end = fit_end + int(self.dataset_lengths[1]*n_entries)
                
                demo_map["fit"][map_key] = entries[0:fit_end]
                demo_map["validate"][map_key] = entries[fit_end:val_end]
                demo_map["test"][map_key] = entries[val_end:]
        
        if stage in ("fit", "validate") or stage is None: 
            datasets_fit = self._make_dataset(demo_map["fit"], "fit")
            datasets_validate = self._make_dataset(demo_map["validate"], "validate")
            self.train_dataset = ConcatDataset(datasets_fit)
            self.val_dataset = ConcatDataset(datasets_validate)

        if stage == "test" or stage is None: 
            datasets_test = self._make_dataset(demo_map["test"], "test")
            self.test_dataset = ConcatDataset(datasets_test)
            
    def teardown(self, stage: Optional[str]=None) -> None:
        stages_to_clean = []
        if stage == "fit":
            stages_to_clean.extend(["train", "val"])
        elif stage == "validate": 
            stages_to_clean.append("val")
        elif stage == "test":
            stages_to_clean.append("test")
        else: 
            stages_to_clean.extend(["train", "val", "test"])

        for prefix in stages_to_clean:
            attr_name = f"{prefix}_dataset"
            dataset = getattr(self, attr_name, None)
            if isinstance(dataset, ConcatDataset):
                for subset in dataset.datasets:
                    base_set = getattr(subset, "dataset", subset)
                    if hasattr(base_set, "close") and callable(base_set.close):
                        base_set.close()
            setattr(self, attr_name, None)
        
    def __del__(self):
        try:
            self.teardown()
        except Exception:
            pass
        
    def _make_dataset(self, demo_map, stage): 
        dataset = [
            MimicGenRobotDataset(
                demo_map=demo_map[f"{task}{robot}"],
                dataframe_gripper=self.dataframe_gripper, 
                transforms_list=self.transforms_list, 
                contrastive_transforms=None if stage == "test" else self.contrastive_transforms, 
                window_size=None if stage == "test" else self.window_size,
                chunk_size=None if stage == "test" else self.chunk_size, 
                temporal_smoothing=None if stage == "test" else self.temporal_smoothing, 
                positive_window_size=None if stage == "test" else self.positive_window_size, 
                negative_window_size=None if stage == "test" else self.negative_window_size, 
                crop_factor=self.crop_factor,
                noise_level=self.noise_level
            ) 
            for task in self.tasks 
            for robot in self.robots
        ]
            
        return dataset 
        
    def _make_dataloader(self, dataset, shuffle: bool) -> DataLoader:                 
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
                persistent_workers=self.persistent_workers if self.num_workers > 0 else False, 
                collate_fn=collate_discover,
                )
            
        return DataLoader(
            dataset=dataset, 
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,  
            pin_memory=self.pin_memory, 
            persistent_workers=self.persistent_workers if self.num_workers > 0 else False,
            drop_last=self.drop_last,
            collate_fn=collate_discover, 
            )
    
    def train_dataloader(self) -> DataLoader:
        return self._make_dataloader(self.train_dataset, shuffle=self.shuffle)
    
    def val_dataloader(self) -> DataLoader:
        return self._make_dataloader(self.val_dataset, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._make_dataloader(self.test_dataset, shuffle=False)