import os 
import h5py 
import numpy as np 
import pandas as pd 
from typing import  List, Optional, Tuple

import lightning as pl 
from torch.utils.data import Dataset, DataLoader, random_split

from data.discover.dataset import MimicGenRobotDataset

class MimicGenRobotDataModule(pl.LightningDataModule): 
    def __init__(self, 
        data_dir: os.PathLike, # directory containing the hdf5 trajectory files 
        meta_dir: os.PathLike, # directory containing the hdf5 files metadata (e.g. min & max of depth maps)
        window: Optional[int],
        robots: Optional[List[str]], 
        tasks: Optional[List[str]], 
        expand_depth: Optional[str], # grayscale, colormap 
        batch_size: int,
        shuffle: bool,  
        num_workers: int, 
        pin_memory: bool, 
        persistent_workers: bool,
        dataset_lengths: List[int], 
        transforms: List[str],
        *args, **kwargs) -> None:
        super().__init__()
        
        # data kwargs
        self.data_dir = data_dir
        self.meta_dir = meta_dir
        self.window = window
        self.robots = robots 
        self.tasks = tasks 
        self.expand_depth = expand_depth 
        
        # dataloading kwargs
        self.batch_size = batch_size 
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.dataset_lengths = dataset_lengths
        
        # image augmenations 
        self.transforms = transforms

        all_files = [file for file in os.listdir(self.data_dir) if "depth" in file]
        all_robots = list(set(f.split(".")[-2].split("_")[-1] for f in all_files)) # no given robot -> all robots
        all_tasks = list(set(f.split(".")[-2].split("_")[0] for f in all_files)) # no given task -> all tasks
        
        self.robots = self._filtered_or_all(self.robots, all_robots)
        self.tasks  = self._filtered_or_all(self.tasks, all_tasks)
        self.files = [os.path.join(data_dir, file) for file in all_files if (robot in file for robot in self.robots) and (task in file for task in self.tasks)]
        
        self.depths = self._get_depths()
        self.metadata = self._get_metadata()
        self.demo_map = self._get_demomap()
        self.train_dataset, self.val_dataset, self.test_dataset = self.setup()
        
        self.n_samples_per_epoch = len(self.demo_map)
    
        
    def _filtered_or_all(self, selected, available):
        if selected is None:
            return available
        filtered = [x for x in selected if x in available]
        return filtered if filtered else available
    
    def _get_depths(self) -> pd.DataFrame: 
        depth_path = os.path.join(self.meta_dir, "depths.csv")
        if os.path.isfile(depth_path):
            df = pd.read_csv(depth_path)
        return df

    def _get_metadata(self) -> pd.DataFrame: 
        meta_path = os.path.join(self.meta_dir, "meta.csv")
        if os.path.isfile(meta_path):
            df = pd.read_csv(meta_path)
        else:
            df = pd.DataFrame()

        for file in self.files:
            if file in df.columns:
                continue

            file_path = os.path.join(self.data_dir, file)
            with h5py.File(file_path, "r") as hf:
                data = hf["data"]
                demos = [data[demo]["actions"][()].shape[0] for demo in data.keys()]
            df[file] = demos
        df.to_csv(meta_path, index=False)
        return df 
    
    def _get_demomap(self): # -> List[List[os.PathLike, int, int, None]]: 
        H = np.inf # H: min episode duration -> max possible window size
        demo_map = []
        
        for file in self.files: 
            with h5py.File(file, "r") as hf: 
                data = hf["data"]
                for demo in data.keys(): 
                    len_episode = data[demo]["actions"][()].shape[0]
                    if len_episode < H:
                        H = len_episode
                    demo_map.append([file, demo, len_episode])
        
        if H < self.window: 
            print(f"The chosen size of the window is bigger than the smallest episode length! \n \
                  Therefore, the size of the window gets changed from {self.window} to {H}.")
            self.window = H
        
        return demo_map
        
    def setup(self, stage=None) -> Tuple[Dataset, Dataset, Dataset]:
        dataset = MimicGenRobotDataset(
            demo_map=self.demo_map,
            depths=self.depths,
            window=self.window,
            expand_depth=self.expand_depth,
            transforms= self.transforms
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
            dataset=self.test_dataset, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers, 
            pin_memory=self.pin_memory, 
            persistent_workers=self.persistent_workers,
            )
        return test_dataloader