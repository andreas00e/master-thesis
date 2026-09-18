import os 
import random 
import pandas as pd
from pathlib import Path
from typing import List, Optional, Union

import torch
from torch.utils.data import DataLoader, ConcatDataset, random_split

import lightning.pytorch as pl 

from data.utils.load_files import get_files, get_metadata, get_demo_dict
from data.discover.dataset import MimicGenRobotDataset
from data.utils.collate import collate_discover
from data.discover.utils.sampler import SameRobotBatchSampler


class MimicGenRobotDataModule(pl.LightningDataModule): 
    def __init__(self, 
        data_dir: Union[str, os.PathLike], # directory containing the hdf5 trajectory files 
        meta_dir: Union[str, os.PathLike], # directory containing the hdf5 files metadata (e.g. min & max of depth maps)
        robots: Optional[Union[str, List[str]]], 
        tasks: Optional[Union[str, List[str]]], 
        n_ds: int, # d0 or d0 and d1
        f_ds: float, # percentage of dataset use
        depth: bool, 
        crop_factor: float,        
        noise_level: float, 
        window: int, 
        chunk: int, 
        consistent_batch: bool, 
        t_smooth: bool, 
        batch_size: int,
        shuffle: bool,  
        num_workers: int, 
        pin_memory: bool, 
        persistent_workers: bool,
        drop_last: bool, 
        dataset_lengths: List[float],
        seed: int, 
        ctr_transforms: bool, 
        transforms: List[str], 
        ) -> None:
        super().__init__()
        
        if not n_ds in [1, 2]: 
            raise ValueError(f"n_ds has to be 1 or 2, got {n_ds}.")
        if not 0 < f_ds <= 1: 
            raise ValueError(f"f_ds has to be in (0, 1], got {f_ds}.")
        if not 0 < crop_factor <= 1: 
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
        self.consistent_batch = consistent_batch
        self.t_smooth = t_smooth

        # Dataloading kwargs
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.drop_last = drop_last
        self.dataset_lengths = dataset_lengths
        self.seed = seed

        # Image transformations/ augmenations
        self.ctr_transforms = ctr_transforms
        self.transforms = transforms
        
        # File handling 
        self.robots, self.tasks, self.files = get_files(self.data_dir, self.depth, robots, tasks) # all hdf5 files containg given robot(s) and task(s)
        self.metadata = get_metadata(self.meta_dir, self.files)
        self.df_gripper = pd.read_csv(self.meta_dir / "gripper_state_robot.csv") # TODO: Move .csv to config 
        self.demo_map, self.window = get_demo_dict(self.metadata, self.files, self.window)  # Tuple[Dict[str, List[Tuple[str, str, int]]], int]
            
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
            t_smooth=self.t_smooth, 
            crop_factor=self.crop_factor,
            noise_level=self.noise_level,
            ctr_transforms=self.ctr_transforms, 
            transforms=self.transforms
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
                # multiprocessing_context="spawn"  
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
            # multiprocessing_context="spawn"  
            )
    
    def train_dataloader(self) -> DataLoader:
        return self._make_dataloader(self.train_dataset, shuffle=self.shuffle)
    
    def val_dataloader(self) -> DataLoader:
        return self._make_dataloader(self.val_dataset, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._make_dataloader(self.test_dataset, shuffle=False)