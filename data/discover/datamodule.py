import os 
import pandas as pd
from typing import  List, Optional, Tuple, Union

import lightning as pl 
from torch.utils.data import Dataset, DataLoader, random_split

from data.utils.load_files import get_files, get_metadata, get_demomap
from data.discover.dataset import MimicGenRobotDataset

from data.utils.collate import collate_discover


class MimicGenRobotDataModule(pl.LightningDataModule): 
    def __init__(self, 
        data_dir: os.PathLike, # directory containing the hdf5 trajectory files 
        meta_dir: os.PathLike, # directory containing the hdf5 files metadata (e.g. min & max of depth maps)
        window: int,
        chunks: int,
        crop_factor: float,  
        noise_level: float, 
        robots: Optional[Union[str, List]], 
        tasks: Optional[Union[str, List]], 
        depth: bool, 
        batch_size: int,
        shuffle: bool,  
        num_workers: int, 
        pin_memory: bool, 
        # multiprocessing_context: str, 
        persistent_workers: bool,
        dataset_lengths: List[int], 
        transforms: List[str],
        *args, **kwargs) -> None:
        super().__init__()
        
        # Data kwargs
        self.data_dir = data_dir
        self.meta_dir = meta_dir
        self.window = window
        self.chunks = chunks
        self.crop_factor = crop_factor 
        self.noise_level = noise_level
        self.robots = list(robots) if isinstance(robots, str) else robots
        self.tasks = list(tasks) if isinstance(robots, str) else tasks
        self.depth = depth
        
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
        self.files = get_files(self.data_dir, self.depth, self.robots, self.tasks)
        self.meta_data = get_metadata(self.meta_dir, self.files)
        self.df_gripper = pd.read_csv(os.path.join(self.meta_dir, "gripper_state_robot.csv"))
        self.demo_map, self.window = get_demomap(self.meta_data, self.files, self.window)
        
        print(f"Number of demos in demo_map: {len(self.demo_map)}")
    
        self.dataset_, self.train_dataset, self.val_dataset, self.test_dataset = [None]*4 
    
    def teardown(self, stage=None) -> None:
        dataset_ = getattr(self, "dataset_", None)
        
        if dataset_ is None: 
            return 
        if not (hasattr(dataset_, "close") and callable(dataset_.close)): 
            return      
        try: 
            dataset_.close()
        except Exception: 
            if getattr(self, "trainer", None) is not None: 
                logger = getattr(self.trainer, "logger", None)
                if logger is not None: 
                    step = getattr(self.trainer, "global_step", 0)
                    logger.log_metrics({"datamodule/teardown_error": 1.0}, step=step)        
        
        self.dataset_ = None 
        self.train_dataset = None 
        self.val_dataset = None 
        self.test_dataset = None 
       
    def setup(self, stage=None) -> None:
        dataset = MimicGenRobotDataset(
            demo_map=self.demo_map[:1000],
            df_g=self.df_gripper, 
            window=self.window,
            chunks=self.chunks, 
            crop_factor=self.crop_factor,
            noise_level=self.noise_level,
            transforms=self.transforms
            )
        
        self.dataset_ = dataset
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(dataset, lengths=self.dataset_lengths)
    
    def train_dataloader(self):
        train_dataloader = DataLoader(
            collate_fn=collate_discover,    
            dataset=self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=self.shuffle,    
            num_workers=self.num_workers,  
            pin_memory=self.pin_memory, 
            # multiprocessing_context=self.multiprocessing_context, 
            persistent_workers=self.persistent_workers, 
            )
        return train_dataloader
    
    def val_dataloader(self):
        val_dataloader = DataLoader(
            collate_fn=collate_discover,    
            dataset=self.val_dataset, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers, 
            pin_memory=self.pin_memory, 
            # multiprocessing_context=self.multiprocessing_context, 
            persistent_workers=self.persistent_workers, 
            )
        return val_dataloader # 1
    
    def test_dataloader(self):
        test_dataloader = DataLoader(
            collate_fn=collate_discover,    
            dataset=self.test_dataset, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers, 
            pin_memory=self.pin_memory, 
            # multiprocessing_context=self.multiprocessing_context, 
            persistent_workers=self.persistent_workers, 
            )
        return test_dataloader