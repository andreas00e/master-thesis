import math
import random
from typing import Dict, Iterator, List

from torch.utils.data import ConcatDataset, Sampler


class SameRobotBatchSampler(Sampler):
    def __init__(
        self, 
        concat_dataset: ConcatDataset, 
        n_robots: int,
        batch_size: int,
        shuffle: bool=True, 
        drop_last: bool=False, 
        seed: int=42, 
        ) -> None:        
        self.concat_dataset = concat_dataset
        self.n_robots = n_robots
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0 
                
        self.dataset_indices: Dict[int, List] = {robot_index: [] for robot_index in range(self.n_robots)} # stores idxs of dataset items of one robot in global concat dataset
        
        start_idx = 0
        for dataset_idx, dataset_size in enumerate(self.concat_dataset.cumulative_sizes):
            robot_id = dataset_idx % n_robots
            self.dataset_indices[robot_id].extend(range(start_idx, dataset_size)) # idxs of dataset items of one robot in global concat dataset
            start_idx = dataset_size
    
    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self) -> Iterator[List[int]]:
        rng = random.Random(self.seed + self.epoch)
        batches = []

        for indices in self.dataset_indices.values(): # idxs of dataset items of one robot (multiple tasks and distributions) in global concat dataset
            if self.shuffle:
                indices = indices.copy()
                rng.shuffle(indices)
            
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i:i+self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    batches.append(batch)
        
        rng.shuffle(batches)
            
        yield from batches

    def __len__(self) -> int:
        total_batches = 0
        for indices in self.dataset_indices.values():
            n = len(indices) # number of items for one robot in global concat dataset 
            if self.drop_last:
                total_batches += n // self.batch_size
            else:
                total_batches += math.ceil(n / self.batch_size) # Exact batch count including partials    
            
        return total_batches