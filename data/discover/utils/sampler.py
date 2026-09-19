import math
import random
from typing import Iterator, List

import torch 
from torch.utils.data import Sampler


class SameRobotBatchSampler(Sampler):
    def __init__(
        self, 
        concat_dataset: torch.utils.data.ConcatDataset, 
        n_robots: int,
        batch_size: int,
        shuffle=True, 
        drop_last=False
        ) -> None:
        
        self.concat_dataset = concat_dataset
        self.n_robots = n_robots
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
                
        self.dataset_indices = {robot_index: [] for robot_index in range(n_robots)}
        
        start_idx = 0
        for dataset_index, dataset_size in enumerate(self.concat_dataset.cumulative_sizes):
            robot_id = dataset_index % n_robots
            self.dataset_indices[robot_id].extend(list(range(start_idx, dataset_size)))            
            start_idx = dataset_size

    def __iter__(self) -> Iterator[List[int]]:
        batches = []
        
        for indices in self.dataset_indices.values():
            if self.shuffle:
                indices = indices.copy()
                random.shuffle(indices)
            
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i:i+self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    batches.append(batch)
        
        if self.shuffle:
            random.shuffle(batches)
            
        for batch in batches:
            yield batch

    def __len__(self) -> int:
        total_batches = 0
        for indices in self.dataset_indices.values():
            n = len(indices)
            if self.drop_last:
                total_batches += n // self.batch_size
            else:
                total_batches += math.ceil(n / self.batch_size)  # Exact batch count including partials                
        
        return total_batches