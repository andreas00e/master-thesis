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

from data.mimic.instructions import stack_instructions

class MimicgenDataset(Dataset):
    def __init__(self, load_dir: Path, robot: Union[List, str], task: Union[List, str], action_horizon: int, image_horizon: Optional[int]=None, view: Optional[str]=None, transform: Optional[bool]=True):
        super().__init__()
        self.load_dir = load_dir
        self.robot = robot 
        self.task = task 
        self.view = view 
        self.action_horizon = action_horizon 
        self.image_horizon = image_horizon # 
        
        self.demo_count = dict()
        self.files, self.traj_map = [], []
        self.instructions = stack_instructions 
        
        # Most of transformations are handled in the r3m forward() function 
        # See at: https://github.com/facebookresearch/r3m/blob/main/r3m/models/models_r3m.py
        if transform: 
            self.transform = v2.Compose([ 
                v2.ToImage(), # (H, W, C) -> (C, H, W), dtype uint
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
            if index+self.action_horizon <= actions.shape[0]: # (actions.shape=(episode length, action_dim)) # Go to the bound due to slicing 
                # If within bounds, slice the actions array to get a trajectory of length 'horizon'
                actions = actions[index:index+self.action_horizon, :] 
                
            else: 
                overhead = (index+self.action_horizon)-actions.shape[0] # If out of bounds, calculate the overhead (number of missing steps)
                padding = np.zeros((overhead, actions.shape[1])) # Create a zero-padded array to fill the missing steps
                actions = np.vstack((actions[index:, :], padding)) # Combine actions and zero padding 
                
            actions = torch.from_numpy(actions).float()
            
            # Used for original HDF5 file without depth information          
            # obs =  data['obs'] # TASK OBSERVATIONS (containing all available RGB views)
            
            obs = data['next_obs'] # TASK OBSERVATIONS (containing all available RGB views and depth maps)
            
            rgb_views, d_views = [], []
            for key in obs.keys(): # hf['data']['demo_index']['next_obs]['agentview_depth' | 'agentview_image' | 'robot0_eye_in_hand_depth' | 'robot0_eye_in_hand_image']
                if 'image' in key: # Views of RGB images
                    rgb_views.append(key) 
                elif 'depth' in key: 
                    d_views.append(key) # Views of depth maps 
                            
            images = dict()
            for rgb_view, d_view in zip(rgb_views, d_views): # Iterate over all available zipped rgb and depth views (mimicgen core dataset: 'agentview', 'robot0_eye_in_hand')
                rgb_images = obs[rgb_view][()] # Current rgb view observations (episode_length, H, W, C=3)
                d_images = obs[d_view][()] # Current depth view observations (episode_length, H, W, C=1)
                frames = np.concatenate([rgb_images, d_images],  axis=-1) # Concatenate images along channel dimension -> (episode_length, H, W, '(C=3)+(C=1)')
                
                if self.image_horizon < 0: # Input is a sequence of images per iteration
                    raise ValueError("Image horizon cannot be negative!")
                elif self.image_horizon > rgb_images.shape[0]: 
                    raise ValueError("Image horizon cannot be bigger than episode length!")
                
                if self.image_horizon: # We process a series of images instead of a single input image 
                    if index >= self.image_horizon: # The current index is bigger or equal than the specified image horizon -> We don't need any frame padding 
                        frames = frames[index-self.image_horizon:index, ...] # (episode_length, H, W, C) -> (self.image_horizon, H, W, C)
                    else: # The current index is smaller than the specified image horizon (We are at the first steps of the episode)-> We need frame padding
                        image_overhead = self.image_horizon - index # Calculate number of images need for frame padding
                        frame_padding = np.repeat(a=np.expand_dims(frames[index, ...], axis=0), repeats=image_overhead, axis=0) # (image_overhead, H, W, C)
                        new_frames = frames[:index, ...] # (index, H, W, C)
                        frames = np.concatenate((frame_padding, new_frames), axis=0) # -> (self.image_horizon, H, W, C)
                    frames = np.transpose(a=frames, axes=(0, 3, 1, 2)) # (self.image_horizon, H, W, C=4) -> (self.image_horizon, C=4, H, W)
                    
                else: # self.image_horizon = None -> Input is one image per iteration
                    frames = frames[index, ...] # (H, W, C) 
                    frames = np.transpose(a=frames, axes=(2, 0, 1)) # (H, W, #views*C) -> (#views*C, H, W)
                frames = torch.from_numpy(frames).float() # Convert to torch float32
                view = '_'.join(rgb_view.split('_')[:-1])
                images[view] = frames # Update dictionary 
            
                # images = {view: obs[view][()][index] for view in obs.keys() if 'image' in view} # Include all available camera views ('agentview_image', robot0_eye_in_hand_image)
                # if self.transform is not None: 
                #     images = [self.transform(image.copy()) for image in images.values()]    
                # images = torch.concatenate(tensors=images) # concatenated torch tensors (#images*C, 64, 64)
        
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            # ROTATE BOTH IMAGES IN thesis/run.py
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            
            # Find random integer to access one of the stored language instructions at random 
            # rand_int = random.randint(a=0, b=len(self.instructions)-1) 
            i_idx = int(demo.split('_')[-1]) % len(self.instructions)
            instructions = self.instructions[i_idx] # LANGUAGE INSTRUCTION 

        return {
            'actions': actions, # EEF ACTIONS       
            'image': images, # TASK OBSERVATIONS (dict)
            'text': instructions # LANGUAGE INSTRUCTION 
        }
    
        
def main(): 
    args = dict()
    load_dir = '~/ehrensberger/mimicgen/datasets/core/'
    load_dir = os.path.expanduser(load_dir)
    robot = '' # no robot specified for mimicgen core dataset
    task = ['stack_d0_depth.hdf5', 'stack_d1_depth.hdf5'] 
    # view = 'robot0_eye_in_hand_image' # Not neccessary as long as we include all (two) views
    action_horizon = 16
    image_horizon = 10

    args['load_dir'] = load_dir
    args['robot'] = robot
    args['task'] = task
    # args['view'] = view
    args['action_horizon'] = action_horizon
    args['image_horizon'] = image_horizon
    args['transform'] = True

    dataset = MimicgenDataset(**args)
    images = next(iter(dataset))['image']
    
    print(type(images))
    print(images.keys())
    
    print(images['agentview'].shape) # torch.Size([10, 4, 84, 84])
    print(images['robot0_eye_in_hand'].shape) # torch.Size([10, 4, 84, 84])
    
    # max = 1 
    # min = -1 
    
    
    # for i in range(len(dataset)):
    #     action = dataset.__getitem__(i)['action']
    #     if torch.min(action) < min: 
    #         print(torch.min(action))
    #     if torch.max(action) > max: 
    #         print(torch.max(action))
    
    
    # print(dataset.traj_map[1000:1010])
    # exit()
    
    # for i, _ in tqdm.tqdm(enumerate(dataset)): 
    #    item = dataset.__getitem__(i)

    # image = next(iter(dataset))['image']
    # print(torch.max(image))
    # print(torch.min(image))
    # plt.imsave(fname='agentview.png', arr=torch.permute(image[:3, ...] / 255, dims=(1, 2, 0)).numpy())
    # plt.imsave(fname='robot0_eye_in_hand.png', arr=torch.permute(image[3:, ...] / 255, dims=(1, 2, 0)).numpy())
    
    
if __name__ == '__main__': 
    main() 
