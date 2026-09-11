import os 
import h5py 
import random 

from typing import List 

import torch
from torch.utils.data import Dataset 

# from data.discover.utils. import tranforms
from models.discover.tse import TSE 

class TestDataset(Dataset): 
    def __init__(
        self, 
        data_dir, 
        robots: List[str], 
        tasks: List[str]
    ):
        super().__init__()
  
        all_files = [file for file in os.listdir(data_dir) if "depth" not in file and any(robot in file for robot in robots) and any(task in file for task in tasks)] 
        random.shuffle(all_files)

        files = [
            next((f for f in all_files if robot in f and task in f), None)
            for robot in robots
            for task in tasks
        ]
                    
        print(files)
                
    def __len__(self): 
         return 1 
    
    def __getitem__(self, idx=None):        
        idx_file = torch.randint(0, len(self.files), (1, )).item()
        file = self.files[idx_file]
        
        with h5py.File(file, "r") as hf: 
            data = hf["data"]
            demos = data.keys().to_list() 
            idx_demo = torch.randint(0, len(demos), (1, )).item()
            demo = demos[idx_demo]
            
            rgb_obs = data[demo]["obs"]["robot0_eye_in_hand_image"]
            rgb_obs = torch.from_numpy(rgb_obs).permute(0, 3, 1, 2)
            
            return rgb_obs 
            
def main():
    checkpoint_path = "/home/bing_TUM/ehrensberger/master-thesis/"
    data_dir = "/home/bing_TUM/ehrensberger/master-thesis/imports/mimicgen/datasets/robot/"
    robots = ["iiwa", "panda", "sawyer", "ur5e"]
    tasks = ["square", "threading"]
    
    dataset_kwargs = {
        "data_dir": data_dir, 
        "robots": robots, 
        "tasks": tasks 
    } 
    
    dataset = TestDataset(**dataset_kwargs)

if __name__ == "__main__":
    main() 