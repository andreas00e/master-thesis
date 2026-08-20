import os 
import h5py 

import torch
from torch.utils.data import Dataset 

from data.discover.utils import tranforms

from models.discover.tse import TSE 

class TestDataset(Dataset): 
    def __init__(
        self, 
        data_dir, 
        robot, 
        task, 
        transforms_list
        ):
        super().__init__()
        
        self.files = [file for file in os.list(dir) if robot in file and task in file]
        self.transforms = tranforms(transforms_list)
        
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
    data_dir = "/home/bing_TUM/ehrensberger/master-thesis/imports/mimicgen/datasets/robot"
    robot = "panda"
    task = "threading"
    
    dataset_kwargs = {
        "data_dir": data_dir, 
        "robot": robot, 
        "task": task 
    }
    
    
    dataset = TestDataset(**dataset_kwargs)
    model = TSE.load_from_checkpoint(checkpoint_path)
    model.eval() 
    
    data = next(iter(dataset))
    
    out = model.self(data)
   
    print(out)
     

if __name__ == "__main__":
    main() 