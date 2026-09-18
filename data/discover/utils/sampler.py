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
            self.dataset_indices[robot_id] = list(range(start_idx, dataset_size))
            start_idx = dataset_size
