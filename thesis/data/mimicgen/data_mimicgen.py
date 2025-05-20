import os 
import glob 
import h5py
import random
import numpy as np 
from typing import List, Union 
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2

from thesis.data.mimicgen.instructions import stack_instructions


class MimicgenDataset(Dataset):
    def __init__(self, load_dir: Path, robot: Union[List, str], task: Union[List, str], view: str, horizon: int, transform: bool=True):
        super().__init__()
        self.load_dir = load_dir
        self.robot = robot 
        self.task = task 
        self.view = view 
        self.horizon = horizon 
        
        self.files, self.traj_map = [], []
        self.instructions = stack_instructions
        
        # Think about 
        self.transform = v2.Compose([ 
            v2.ToImage(), # Convert to (C, H, W), dtype uint8
            v2.ToDtype(torch.float32, scale=True), # Convert to float32 and divide by 255
            v2.Resize((64, 64)), # Resize
            v2.Normalize(mean=[0.485, 0.456, 0.406], # Normalize with standard ImageNet normalization parameters
                        std=[0.229, 0.224, 0.225]), 
            ])  
         
        # Check if one robot/task (str) or multiple robots/tasks (List[str]) have been passed
        # One robot, one task  
        if isinstance(self.robot, str) and isinstance(self.task, str): 
            self.files = glob.glob(os.path.join(self.load_dir, f"*{self.task}*{self.robot}*"))
        # TODO: Test and revise the following conditional block
        # One robot, multiple tasks 
        elif isinstance(self.robot, str) and isinstance(self.task, list):
            assert len(self.task) > 0, "No task has been passed."
            self.files = [glob.glob(os.path.join(self.load_dir, task))[0] for task in self.task]
        # Multiple robots, one task 
        elif isinstance(self.robot, list) and isinstance(self.task, str): 
            assert len(self.robot) > 0, "No robot has been passed."
            self.files = [file for file in glob(self.load_dir+self.task) if any(robot in file for robot in self.robot)]    
        # Multiple robots, multiple tasks 
        elif isinstance(self.robot, list) and isinstance(self.task, list): 
            assert len(self.robot) > 0, "No robot has been passed."
            assert len(self.task) > 0, "No task has been passed."
            self.files = [file for file in glob.glob(self.load_dir) if any(robot in file for robot in self.robots) and any(task in file for task in self.task)]
        else: 
            raise TypeError("\"robot\" has to be either a str or a list.")
        
        for file in self.files: 
            # Open current hdf5 file
            with h5py.File(file, 'r') as hf:
                data = hf['data']
                # Open current demo
                for demo in data.keys(): 
                    actions = data[demo]['actions'][()]
                    # Create trajectory map for every time step of current demo to enable later indexed data loading 
                    self.traj_map.extend((file, demo, i) for i in range(actions.shape[0]))
                
    def __len__(self): 
        return len(self.traj_map)          

    def __getitem__(self, index):
        input = self.traj_map[index] # (file, demo, index)
        return self.load_trajectory(*input) 
    
    # Actions are already normalized accorind to: TODO: Add link to mimicgen page
    def load_trajectory(self, file, demo, index): 
        with h5py.File(file, 'r') as hf: 
            actions = hf['data'][demo]['actions'][()] # EEF ACTION
            # Check if the current index + horizon is within the bounds of the actions array
            if index+self.horizon < actions.shape[0]: 
                # If within bounds, slice the actions array to get a trajectory of length 'horizon'
                actions = actions[index:index+self.horizon, :] 
            else: 
                # If out of bounds, calculate the overhead (number of missing steps)
                overhead = np.abs(actions.shape[0]-(index+self.horizon))
                # Create a zero-padded array to fill the missing steps
                padding = np.zeros((overhead, actions.shape[1]))
                actions = np.vstack((actions[index:, :], padding))
            
            obs = hf['data'][demo]['obs']
            # Include all available camera views ('agentview_image', robot0_eye_in_hand_image)
            images = [obs[view][()][index] for view in obs.keys() if 'image' in view] # TASK OBSERVATION
            
            # Find random integer to access one of the stored language instructions at random 
            rand_int = random.randint(a=0, b=len(self.instructions)-1) # LANGUAGE INSTRUCTION 

        return {
            'actions': torch.from_numpy(actions).float(),            
            'image': torch.cat([self.transform(image) for image in images]), # concatenated torch tensors (#immages*C, 64, 64)
            'text': self.instructions[rand_int]
        }
    
        
def main(): 
    args = dict()
    load_dir = '~/ehrensberger/mimicgen/datasets/core/'
    load_dir = os.path.expanduser(load_dir)
    robot = '' # no robot specified for mimicgen core dataset
    task = ['stack_d0.hdf5', 'stack_d1.hdf5'] 
    view = 'robot0_eye_in_hand_image'
    horizon = 16

    args['load_dir'] = load_dir
    args['robot'] = robot
    args['task'] = task
    args['view'] = view
    args['horizon'] = horizon
    args['transform'] = True

    dataset = MimicgenDataset(**args)
    print(len(dataset))
    
    image = next(iter(dataset))['image']
    print(image.shape)
    

if __name__ == '__main__': 
    main() 
