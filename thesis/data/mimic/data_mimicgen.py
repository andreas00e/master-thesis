import os 
import cv2
import glob 
import h5py
import numpy as np 
from typing import List, Optional, Union 
from pathlib import Path

import torch
from torch.utils.data import Dataset

from thesis.data.mimic.language_instructions import stack_instructions

def filter_recursion(task: str, i=1):
    task = task.split('_')
    if 'd' in task[i]: 
        if i == 1:
            return task[0]
        else: 
            return '_'.join(task[:i])
    else: 
        task = '_'.join(task)
        i += 1
        return filter_recursion(task, i)

def filter_task(tasks: List[str]): 
    tasks = ['stack_d0_depth.hdf5', 'stack_d1_depth.hdf5']
    reduced_tasks = []

    
    for task in tasks:
        reduced_tasks.append(filter_recursion(task))
    return reduced_tasks

class MimicgenDataset(Dataset):
    def __init__(self, load_dir: Path, robot: Union[List[str], str, None], task: Union[List, str], action_horizon: int, image_horizon: Union[int, None], observations: str, expand_depth: str):
        super().__init__()
        self.load_dir = load_dir
        self.robot = robot 
        self.task = task 
        self.action_horizon = action_horizon
        self.image_horizon = image_horizon 
        
        self.observations = observations # 'forward', 'backward', 'single'
        self.expand_depth = expand_depth # 'grayscale', 'colormap'
        
        self.demo_count = dict()
        self.files, self.traj_map = [], []
        self.rgb_views, self.d_views = [], []
        self.instructions = stack_instructions 
    
        # Check if one robot/task (str) or multiple robots/tasks (list[str]) have been passed 
        if (isinstance(self.robot, str) or isinstance(self.robot, type(None))) and isinstance(self.task, str): # One robot, one task  #TODO: Check implementation 
            self.files = glob.glob(os.path.join(self.load_dir, f"*{self.task}*{self.robot}*"))
        elif (isinstance(self.robot, str) or isinstance(self.robot, type(None))) and isinstance(self.task, list): # One robot, multiple tasks 
            assert len(self.task) > 0, "No task has been passed!"
            self.files = [glob.glob(os.path.join(self.load_dir, task))[0] for task in self.task]
        elif isinstance(self.robot, list) and isinstance(self.task, str): # Multiple robots, one task #TODO: Check implementation 
            assert len(self.robot) > 0, "No robot has been passed."
            self.files = [file for file in glob(self.load_dir+self.task) if any(robot in file for robot in self.robot)]    
        elif isinstance(self.robot, list) and isinstance(self.task, list): # Multiple robots, multiple tasks #TODO: Check implementation 
            assert len(self.robot) > 0, "No robot has been passed."
            assert len(self.task) > 0, "No task has been passed."
            self.files = [file for file in glob.glob(self.load_dir) if any(robot in file for robot in self.robots) and any(task in file for task in self.task)]
        else: 
            raise TypeError("\"robot\" has to be either a str or a list.")
        
        for file in self.files: 
            with h5py.File(file, 'r') as hf: 
                data = hf['data']
                demos = data.keys() # list of demos (usually 1000 per hdf5 file)
                self.demo_count[file] = len(demos)
                for demo in demos: 
                    actions = data[demo]['actions'][()]
                    self.traj_map.extend((file, demo, index) for index in range(actions.shape[0])) # Create trajectory map for every time step of current demo to enable later indexed data loading 

                obs = data[demo]['next_obs'] # Collect available views  
                for key in obs.keys(): 
                    if 'image' in key: # Views of RGB images
                        if key not in self.rgb_views: 
                            self.rgb_views.append(key) 
                    elif 'depth' in key: # Views of depth maps
                        if key not in self.d_views: 
                            self.d_views.append(key)  
                
    def __len__(self): 
        return len(self.traj_map)          

    def __getitem__(self, idx):
        input = self.traj_map[idx] # (file, demo, index)
        return self.load_trajectory(*input) 
    
    # 'Actions should be normalized between -1 and 1': https://robomimic.github.io/docs/datasets/overview.html
    def load_trajectory(self, file, demo, index):
        with h5py.File(file, 'r') as hf:
            data = hf['data'][demo] 
            actions = data['actions'][()] # EEF ACTION (episode_length, action_dim) 
            obs = data['next_obs'] # TASK OBSERVATIONS (containing all available RGB views and depth maps)    
            images = dict()    
            
            episode_length = actions.shape[0]      

            if index+self.action_horizon <= episode_length: # Check if current index + horizon is within bounds of actions array
                actions = actions[index:index+self.action_horizon, :] # If within bounds, slice actions array to get trajectory of length 'horizon'          
            else: 
                actions_overhead = (index+self.action_horizon)-actions.shape[0] # If out of bounds, calculate overhead (number of missing steps)
                action_padding = np.zeros((actions_overhead, actions.shape[1])) # Create zero-padded array to fill missing steps
                actions = np.vstack((actions[index:, :], action_padding)) # Combine actions and zero padding         
            actions = torch.from_numpy(actions).float()
            
            for rgb_view, d_view in zip(self.rgb_views, self.d_views): # Iterate over all available zipped rgb and depth views (mimicgen core dataset: 'agentview', 'robot0_eye_in_hand')
                rgb_images = obs[rgb_view][()] # Current rgb view observations (episode_length, H, W, C=3)
                d_images = obs[d_view][()] # Current depth view observations (episode_length, H, W, C=1)
                
                # Depth map normalization to [0, 1]
                if d_view == 'robot0_eye_in_hand_depth': 
                    min = 0.04186 # Values were found by iterating over the dataset and including values that include ~90% of the distribution
                    max = 2.5    
                elif d_view == 'agentview_depth': 
                    min = 0.3
                    max = 3.2
                    
                frames = np.concatenate([rgb_images, d_images],  axis=-1) # Concatenate images along channel dimension -> (episode_length, H, W, C=4)
                
                if self.observations == 'single': # Input is one image per time step
                    frames = frames[index:index+1, ...]

                elif self.observations == 'forward': # Same logic as for actions
                    assert self.image_horizon <= self.action_horizon, 'Case where iamge horizon is bigger than action horizon is not accounted for!'
                    if index+self.image_horizon <= episode_length: 
                        frames = frames[index:index+self.image_horizon, ...]
                    else: 
                        actions_overhead = (index+self.image_horizon)-frames.shape[0] 
                        frame_padding = np.repeat(a=np.expand_dims(frames[-1, ...], axis=0), repeats=actions_overhead, axis=0) # Repeat last image to filling missing steps
                        frames = np.vstack((frames[index:, ...], frame_padding)) 
                    
                elif self.observations == 'backward': # We process a series of images instead of a single input image 
                    assert self.image_horizon > 0, 'Image horizon has to be bigger than zero!'
                    if index >= self.image_horizon : # The current index is bigger or equal than the specified image_horizon -> We don't need any frame padding 
                        frames = frames[index-self.image_horizon:index, ...] # (episode_length, H, W, C) -> (self.image_horizon, H, W, C)
                    else: # The current index is smaller than the specified image horizon (We are at the first steps of the episode)-> We need frame padding
                        image_overhead = self.image_horizon-index # Calculate number of images need for frame padding
                        frame_padding = np.repeat(a=np.expand_dims(frames[index, ...], axis=0), repeats=image_overhead, axis=0) # (image_overhead, H, W, C)
                        new_frames = frames[:index, ...] # (index, H, W, C)
                        frames = np.vstack((frame_padding, new_frames)) # -> (self.image_horizon, H, W, C)
                    
                frames = np.transpose(a=frames, axes=(0, 3, 1, 2)) # (horizon/self.image_horizon, H, W, C=6) -> (horizon/self.image_horizon, C=6, H, W)
                frames[:, 3, ...] = (frames[:, 3, ...]-min)/(max-min) # Normaize depth map to ~[0, 1]
                frames[:, 3, ...] = np.clip(frames[:, 3, ...], a_min=0.0, a_max=1.0) # Clip outliers
                frames[:, 3, ...] *= 255.0
                 
                if self.expand_depth: 
                    new_frames = np.zeros((frames.shape[0], 6, *frames.shape[-2:]))
                    new_frames[:, :3, ...] = frames[:, :3, ...] # copy rgb part 
                    if self.expand_depth == 'colormap': # convert gray scale image to colorized image through colormap 
                        depth_frames = np.stack([cv2.cvtColor(cv2.applyColorMap(frames[i, 3, ...].astype(np.uint8), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB) for i in range(frames.shape[0])])
                        depth_frames = np.transpose(depth_frames, axes=(0, 3, 1, 2))
                    elif self.expand_depth == 'grayscale': # repeat depth channel three times
                        depth_frames = frames[:, 3:3+1, ...]
                        depth_frames = np.repeat(depth_frames, repeats=3, axis=1) # (episode_length, C=3, H, W)
                        
                    new_frames[:, 3:, ...] = depth_frames
                    frames = new_frames

                frames = torch.from_numpy(frames).float()
                view = '_'.join(rgb_view.split('_')[:-1])
                images[view] = frames 
            
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            # ROTATE BOTH IMAGES IN thesis/run.py
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            
            # Find random integer to access one of the stored language instructions at random 
            # rand_int = random.randint(a=0, b=len(self.instructions)-1) # same demo get's different language instruction 
            i_idx = int(demo.split('_')[-1]) % len(self.instructions) # same demo always get's same language instruction
            instructions = self.instructions[i_idx] # LANGUAGE INSTRUCTION 

        return {
            'actions': actions, # EEF ACTIONS       
            'image': images, # TASK OBSERVATIONS (dict)
            'text': instructions # LANGUAGE INSTRUCTION 
        }
    