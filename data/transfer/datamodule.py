import os 
from omegaconf import OmegaConf
from typing import  List, Optional, Tuple

import torch
import lightning as pl 
from torch.utils.data import Dataset, DataLoader, random_split

from data.utils.load_files import get_files, get_metadata, get_demomap
from data.utils.collate import collate_transfer
from data.transfer.dataset import TransferDataset


class TransferDataModule(pl.LightningDataModule): 
    def __init__(self, 
        data_dir: os.PathLike, # directory containing the hdf5 trajectory files 
        meta_dir: os.PathLike, # directory containing the hdf5 files metadata (e.g. min & max of depth maps)        cfgs_dir: os.PathLike, 
        cfgs_dir: os.PathLike, 
        horizon: int,
        crop_factor: float, 
        robots: Optional[List[str]], 
        tasks: Optional[List[str]], 
        depth: bool, 
        batch_size: int,
        shuffle: bool,  
        num_workers: int, 
        pin_memory: bool,
        drop_last: bool,  
        persistent_workers: bool,
        dataset_lengths: List[int], 
        transforms: List[str],
        *args, **kwargs) -> None:
        super().__init__()
        
        # Data kwargs
        self.data_dir = data_dir
        self.meta_dir = meta_dir
        self.cfgs_dir = cfgs_dir
        self.horizon = horizon
        self.crop_factor = crop_factor
        self.robots = robots 
        self.tasks = tasks 
        self.depth = depth
        
        # Dataloading kwargs
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.persistent_workers = persistent_workers
        self.dataset_lengths = dataset_lengths
        self.transforms = transforms 

        # File handling 
        self.files = get_files(self.data_dir, self.depth, self.robots, self.tasks)
        self.meta_data = get_metadata(self.meta_dir, self.files)

        self.demo_map, _ = get_demomap(self.meta_data, self.files, 8)
        
        self.joint_dsc = self._get_joint_dsc()
        self.train_dataset, self.val_dataset, self.test_dataset = self.setup()
        
    def setup(self, stage=None) -> Tuple[Dataset, ...]:
        dataset = TransferDataset(
            demo_map=self.demo_map,
            horizon=self.horizon,
            crop_factor=self.crop_factor,
            depth=self.depth,
            joint_dsc=self.joint_dsc,
            transforms=self.transforms
            )
        
        train_dataset, val_dataset, test_dataset = random_split(dataset, lengths=self.dataset_lengths)
        return train_dataset, val_dataset, test_dataset
    
    def _get_joint_dsc(self): 
        cfgs = [os.path.join(self.cfgs_dir, cfg) for cfg in os.listdir(self.cfgs_dir)] 
        
        joint_dsc = {}
        for cfg in cfgs: 
            cfg = OmegaConf.load(cfg)
            robot = next(iter(cfg.keys()))
            values = list(cfg[robot].values())
            values = torch.tensor(values)
            
            joint_dsc[robot] = values
        
        return joint_dsc     
    
    def train_dataloader(self):
        train_dataloader = DataLoader(
            dataset=self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=self.shuffle,    
            num_workers=self.num_workers,  
            collate_fn=collate_transfer, 
            pin_memory=self.pin_memory, 
            persistent_workers=self.persistent_workers, 
            )
        return train_dataloader
    
    def val_dataloader(self):
        val_dataloader = DataLoader(
            dataset=self.val_dataset, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers, 
            collate_fn=collate_transfer, 
            pin_memory=self.pin_memory, 
            persistent_workers=self.persistent_workers, 
            )
        return val_dataloader
    
    def test_dataloader(self):
        test_dataloader = DataLoader(
            dataset=self.test_dataset,             
            batch_size=self.batch_size, 
            num_workers=self.num_workers, 
            collate_fn=collate_transfer, 
            pin_memory=self.pin_memory, 
            persistent_workers=self.persistent_workers, 
            )
        return test_dataloader