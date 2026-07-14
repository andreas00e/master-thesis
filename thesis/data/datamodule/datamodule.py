import os 
import h5py 
import numpy as np 
import pandas as pd 
from typing import List, Optional, Tuple

import lightning as pl 
from torch.utils.data import DataLoader, random_split

from data.dataset.dataset import MimicGenRobotDataset

class MimicGenRobotDataModule(pl.LightningDataModule): 
    def __init__(self, 
        data_dir: os.PathLike, # directory containing the hdf5 trajectory files 
        meta_dir: os.PathLike, # directory containing the hdf5 files metadata (e.g. min & max of depth maps)
        window: Optional[int] = 16, # number of steps looking into the future from current time-steps
        robots: Optional[List[str]] = None, 
        tasks: Optional[List[str]] = None, 
        depth: Optional[bool] = True, # if depth maps are included or not  
        expand_depth: Optional[str] = None, # grayscale, colormap 
        batch_size: int = 16, 
        shuffle: bool = False, 
        num_workers: int = 32, 
        pin_memory: bool = True, 
        persisent_workers: bool = True,
        *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        # data kwargs
        self.data_dir = data_dir
        self.meta_dir = meta_dir
        self.window = window
        self.robots = robots 
        self.tasks = tasks 
        self.depth = depth
        self.expand_depth = expand_depth 
        
        # dataloading kwargs
        self.batch_size = batch_size 
        self.shuffle = shuffle 
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persisent_workers = persisent_workers

        all_files = [file for file in os.listdir(self.data_dir) if ("depth" in file) == self.depth]
        all_robots = list(set(f.split(".")[-2].split("_")[-1] for f in all_files)) # no given robot -> all robots
        all_tasks = list(set(f.split(".")[-2].split("_")[0] for f in all_files)) # no given task -> all tasks
        
        self.robots = self._filtered_or_all(self.robots, all_robots)
        self.tasks  = self._filtered_or_all(self.tasks, all_tasks)
        self.files = [os.path.join(data_dir, file) for file in all_files if (robot in file for robot in self.robots) and (task in file for task in self.tasks)]
        
        self.meta_data = self._get_metadata()
        self.demo_map = self._get_demodata()
        self.train_dataset, self.val_dataset = self.setup()
        
        self.n_samples_per_epoch = len(self.demo_map)
        
    def _filtered_or_all(self, selected, available):
        if selected is None:
            return available
        filtered = [x for x in selected if x in available]
        return filtered if filtered else available

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
    
    def _get_demodata(self) -> Tuple[List[List[os.PathLike, int, int, None]], int]: 
        H = np.inf # H: min. episode duration -> max. possible window size
        demo_map = []
        
        for file in self.files: 
            with h5py.File(file, "r") as hf: 
                data = hf["data"]
                for demo in data.keys(): 
                    len_episode = data[demo]["actions"][()].shape[0]
                    if len_episode < H:
                        H = len_episode
                    demo_map.append([file, demo, len_episode, None])
        
        if H > self.window: 
            print(f"The chosen size of the window is bigger than the smallest episode length! \n \
                  Therefore, the size of the window gets changed from {self.window} to {H}.")
            self.window = H
        return demo_map
        
    def setup(self, stage=None) -> Tuple[DataLoader, DataLoader]:
        dataset = MimicGenRobotDataset(
            self.files, 
            self.demo_map,
            self.window, 
            self.depth, 
            self.expand_depth, 
            self.window
            )
        
        train_dataset, val_dataset = random_split(dataset, lengths=self.dataset_lengths)
        return train_dataset, val_dataset
    
    def train_dataloader(self):
        train_dataloader = DataLoader(
            self.train_dataset, 
            self.batch_size, 
            self.shuffle, 
            self.num_workers, 
            self.pin_memory, 
            self.persisent_workers
            )
        return train_dataloader
    
    def val_dataloader(self):
        val_dataloader = DataLoader(
            self.val_dataset, 
            self.batch_size, 
            self.num_workers, 
            self.pin_memory, 
            self.persisent_workers
            )
        return val_dataloader