import os 
import glob 
import h5py
import tqdm 
import random
import numpy as np 
from typing import List, Optional, Union 
from pathlib import Path
from matplotlib import pyplot as plt 

import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2

from thesis.data.mimicgen.instructions import stack_instructions


class MimicgenDataset(Dataset):
    def __init__(self, load_dir: Path, robot: Union[List, str], task: Union[List, str], horizon: int, view: Optional[str]=None, transform: Optional[bool]=True):
        super().__init__()
        self.load_dir = load_dir
        self.robot = robot 
        self.task = task 
        self.view = view 
        self.horizon = horizon 
        
        self.demo_count = dict()
        self.files, self.traj_map = [], []
        self.instructions = stack_instructions 
        
        # Most of transformations are handled in the r3m forward() function 
        # See at: https://github.com/facebookresearch/r3m/blob/main/r3m/models/models_r3m.py
        if transform: 
            self.transform = v2.Compose([ 
                v2.ToImage(), # Convert to (C, H, W), dtype uint
                v2.ToDtype(torch.float32), 
                # v2.ToDtype(torch.float32, scale=False), # Convert to float32 and divide by 255
                # v2.Resize((64, 64)), # Resize
                # v2.Normalize(mean=[0.485, 0.456, 0.406], # Normalize with standard ImageNet normalization parameters
                #             std=[0.229, 0.224, 0.225]), 
                ])
        else:
            self.transform = None 
         
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
            with h5py.File(file, 'r') as hf: # Open current hdf5 file
                data = hf['data'] # Open current demo (Usually 1000 per hdf5 file)
                keys = data.keys()
                self.demo_count[file] = len(keys)
                for demo in keys: 
                    actions = data[demo]['actions'][()]
                    # Create trajectory map for every time step of current demo to enable later indexed data loading 
                    self.traj_map.extend((file, demo, index) for index in range(actions.shape[0]))
                
    def __len__(self): 
        return len(self.traj_map)          

    def __getitem__(self, idx):
        input = self.traj_map[idx] # (file, demo, index)
        return self.load_trajectory(*input) 
    
    # Actions are already normalized accorind to: TODO: Add link to mimicgen page
    def load_trajectory(self, file, demo, index): 
        with h5py.File(file, 'r') as hf:
            data = hf['data'][demo]
            actions = data['actions'][()] # EEF ACTION 
            # print(f"demo: {demo}, index: {index}, shape: {actions.shape[0]}")
            # Check if the current index + horizon is within the bounds of the actions array
            if index+self.horizon <= actions.shape[0]: # (actions.shape=(episode length, action_dim)) # Go to the bound due to slicing 
                # If within bounds, slice the actions array to get a trajectory of length 'horizon'
                actions = actions[index:index+self.horizon, :] 
                
            else: 
                overhead = (index+self.horizon)-actions.shape[0] # If out of bounds, calculate the overhead (number of missing steps)
                padding = np.zeros((overhead, actions.shape[1])) # Create a zero-padded array to fill the missing steps
                actions = np.vstack((actions[index:, :], padding)) # Combine actions and zero padding 
                
            actions = torch.from_numpy(actions).float()
                        
            obs =  data['obs'] # TASK OBSERVATIONS
            images = {view: obs[view][()][index] for view in obs.keys() if 'image' in view} # Include all available camera views ('agentview_image', robot0_eye_in_hand_image)

            # rotate robot0_eye_in_hand_image 180 degrees to allgin dataset with robosuite renderer representation
            # images['robot0_eye_in_hand_image'] = np.rot90(images['robot0_eye_in_hand_image'], k=2) 
            
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            # ROTATE BOTH IMAGES IN thesis/run.py
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            
            if self.transform is not None: 
                images = [self.transform(image.copy()) for image in images.values()]
                               
            images = torch.cat(tensors=images) # concatenated torch tensors (#immages*C, 64, 64)
             
            # Find random integer to access one of the stored language instructions at random 
            # rand_int = random.randint(a=0, b=len(self.instructions)-1) 
            i_idx = int(demo.split('_')[-1]) % len(self.instructions)
            instructions = self.instructions[i_idx] # LANGUAGE INSTRUCTION 

        return {
            'actions': actions, # EEF ACTIONS       
            'image': images, # TASK OBSERVATIONS
            'text': instructions # LANGUAGE INSTRUCTION 
        }
    
        
def main(): 
    args = dict()
    load_dir = '~/ehrensberger/mimicgen/datasets/core/'
    load_dir = os.path.expanduser(load_dir)
    robot = '' # no robot specified for mimicgen core dataset
    task = ['stack_d0.hdf5', 'stack_d1.hdf5'] 
    # view = 'robot0_eye_in_hand_image' # Not neccessary as long as we include all (two) views
    horizon = 16

    args['load_dir'] = load_dir
    args['robot'] = robot
    args['task'] = task
    # args['view'] = view
    args['horizon'] = horizon
    args['transform'] = True

    dataset = MimicgenDataset(**args)
    print(dataset.traj_map[1000:1010])
    exit()
    
    for i, _ in tqdm.tqdm(enumerate(dataset)): 
        item = dataset.__getitem__(i)

    # image = next(iter(dataset))['image']
    # print(torch.max(image))
    # print(torch.min(image))
    # plt.imsave(fname='agentview.png', arr=torch.permute(image[:3, ...] / 255, dims=(1, 2, 0)).numpy())
    # plt.imsave(fname='robot0_eye_in_hand.png', arr=torch.permute(image[3:, ...] / 255, dims=(1, 2, 0)).numpy())
    
    
if __name__ == '__main__': 
    main() 
