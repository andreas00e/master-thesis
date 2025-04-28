import os 
import glob 
import h5py
import tqdm
import time 
import itertools

import numpy as np 
import multiprocessing as mp

from typing import List, Union 
from pathlib import Path

from torch.utils.data import Dataset

import threading
from concurrent.futures import ThreadPoolExecutor

import torch
from torchvision.transforms import v2

transforms = v2.Compose([
        v2.Resize(size=(64, 64)),
        v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

class MimicgenDataset(Dataset):
    def __init__(self, load_dir: Path, robot: Union[List, str], task: Union[List, str], view: str, horizon: int, overlap: bool):
        super().__init__()
        self.load_dir = load_dir
        self.robot = robot 
        self.task = task 
        self.files, self.traj_map = [], []
        self.view = view 
        self.horizon = horizon 
        self.overlap = overlap
         
        # Check if one robot/task (str) or multiple robots/tasks (List[str]) have been passed
        # One robot, one task  
        if isinstance(self.robot, str) and isinstance(self.task, str): 
            self.files = glob.glob(os.path.join(self.load_dir, f"*{self.task}*{self.robot}*"))
        # TODO: Test and revise the following conditional block
        # One robot, multiple tasks 
        elif isinstance(self.robot, str) and isinstance(self.task, list):
            assert len(self.task) > 0, "No task has been passed."
            self.files = [file for file in glob.glob(os.path.join(self.load_dir, self.robot)) if any(task in file for task in self.task)]
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
            with h5py.File(file, 'r') as hf:
                data = hf['data']
                for demo in data.keys(): 
                    actions = data[demo]['actions'][()]
                    self.traj_map.extend((file, demo, i) for i in range(actions.shape[0]))
                
    def __len__(self): 
        return len(self.traj_map)          

    def __getitem__(self, index):
        input = self.traj_map[index]
        return self.load_trajectory(*input)    
    
    # TODO: Normalize actions!
    def load_trajectory(self, file, demo, index): 
        with h5py.File(file, 'r') as hf: 
            actions = hf['data'][demo]['actions'][()]
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

            image = hf['data'][demo]['obs']['robot0_eye_in_hand_image'][()][index]

        return {
            'actions': transforms(actions.astype(float)), 
            'image': image.astype(float)
        }
    
    def normalize(): 
        pass 

        
def main(): 
    args = dict()
    load_dir = '~/ehrensberger/mimicgen/datasets/core/'
    load_dir = os.path.expanduser(load_dir)
    robot = ''
    task = 'square'
    # ['agentview_image', robot0_eye_in_hand_image']
    view = 'robot0_eye_in_hand_image'
    horizon = 16
    overlap = False

    args['load_dir'] = load_dir
    args['robot'] = robot
    args['task'] = task
    args['view'] = view
    args['horizon'] = horizon
    args['overlap'] = overlap

    dataset = MimicgenDataset(**args)
    print(len(dataset.traj_map))

    # for element in tqdm.tqdm(dataset, colour='green'): 
    #     actions = element['actions']
    #     min = np.min(actions)
    #     max = np.max(actions)
    #     if min < -1 or max > 1:
    #         print(f"min element in current trajectory: {min}, max element in current trajectory: {max}")
    #         print("-----------------------------------------------------------------------------------")


if __name__ == '__main__': 
    main() 
