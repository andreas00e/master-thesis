import os
import glob 
import copy  
import h5py
import tqdm 
import numpy as np
import pickle as pkl 
 
import time 
from functools import wraps

from dataclasses import dataclass, field 

from robonet.datasets.util.metadata_helper import load_metadata
from robonet.datasets.util.hdf5_loader import *

# Adjusted from /robonet/datasets/util/hdf5_loader.py to be pytorch-compatible 
@dataclass
class HParams:
    target_adim: int = 4 
    target_sdim: int = 5
    state_mismatch: str = STATE_MISMATCH.ERROR # TODO make better flag parsing
    action_mismatch: str = ACTION_MISMATCH.ERROR # TODO make better flag parsing
    img_size: list = field(default_factory=lambda: [48, 64])
    cams_to_load: list = field(default_factory=lambda: [0])
    input_autograsp_action: bool = True
    load_annotations: bool = False 
    zero_if_missing_annotation: bool = False # TODO implement error checking here for jagged reading
    load_T: int = 0 

# EEF position/orientation and binary gripper state (open/closed)
STATE = {
    'X': np.inf, 'Y': np.inf, 'Z': np.inf, 'YAW': np.inf, 'GRIPPER': np.inf
    }

# Individual dictionary for every robot to save min and max state, action, and joint values 
ROBOTDATA = {
    'MIN_STATE': copy.copy(STATE), 
    'MAX_STATE': {k: -v for k, v in copy.copy(STATE).items()}, 
    'MIN_ACTION': copy.copy(STATE), # actions are state deltas 
    'MAX_ACTION': {k: -v for k, v in copy.copy(STATE).items()}, 
    'MIN_Q_POS': np.inf, # min global joint position
    'MAX_Q_POS': -np.inf, # max global joint position 
    'MIN_Q_VEL': np.inf, # min global joint velocity 
    'MAX_Q_VEL': -np.inf, # max global joint velocity 
    }

METADATA = {
    'SAWYER': copy.deepcopy(ROBOTDATA), 'WIDOWX': copy.deepcopy(ROBOTDATA), 
    'BAXTER': copy.deepcopy(ROBOTDATA), 'KUKA': copy.deepcopy(ROBOTDATA), 'FRANKA': copy.deepcopy(ROBOTDATA)
    }

#TODO: Implement function that checks whether all values of METADATA exists and have been changed from the given default values 

def time_it(func): 
    @wraps(func)
    def time_it_wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        total_time = end_time - start_time
        print(f"Execution of function {func.__name__} took {total_time:.4f} seconds")
        return result 
    return time_it_wrapper

def collect_metadata(files):
    state_keys = list(STATE.keys())
    
    #TODO: Add functionality to check for changes in exisiting metadata.pkl 
    # Check if metadata has already been saved 
    if os.path.isfile('metadata.pkl'): 
        with open('./metadata.pkl', 'rb') as metadata: 
            metadata = pkl.load(metadata)
            return metadata
    
    for file in tqdm.tqdm(files, desc="Finding min and max values for normalization", colour='green'): 
        with h5py.File(file, 'r') as hf: 
            # print(hf['metadata'].attrs['object_classes'])
            # get current robot name 
            robot = hf['metadata'].attrs['robot'].upper()
            if robot in METADATA.keys(): 
                try: 
                    q_pos = np.array(hf['env']['qpos'])
                    # print(q_pos.shape)
                    if np.min(q_pos) < METADATA[robot]['MIN_Q_POS']:
                        METADATA[robot]['MIN_Q_POS'] = np.min(q_pos)
                    # find max q_pos values 
                    if np.max(q_pos) > METADATA[robot]['MAX_Q_POS']:
                        METADATA[robot]['MAX_Q_POS'] = np.max(q_pos)

                    # find min q_vel values 
                    q_vel = np.array(hf['env']['qvel'])
                    # print(q_vel.shape)
                    if np.min(q_vel) < METADATA[robot]['MIN_Q_VEL']:
                        METADATA[robot]['MIN_Q_VEL'] = np.min(q_vel)
                    # find max q_vel values 
                    if np.max(q_vel) > METADATA[robot]['MAX_Q_VEL']:
                        METADATA[robot]['MAX_Q_VEL'] = np.max(q_vel)

                    state = np.array(hf['env']['state'])
                    # print(state.shape)
                    for i in range(state.shape[1]): 
                        # find min state values
                        if np.min(state[:, i]) < METADATA[robot]['MIN_STATE'][state_keys[i]]:
                            METADATA[robot]['MIN_STATE'][state_keys[i]] = np.min(state[:, i])
                        # find max state values 
                        if np.max(state[:, i]) > METADATA[robot]['MAX_STATE'][state_keys[i]]:
                            METADATA[robot]['MAX_STATE'][state_keys[i]] = np.max(state[:, i])
                    
                    #TODO: Check if two loops can be combined               
                    actions = np.array(hf['policy']['actions'])
                    # print(actions.shape)
                    for i in range(actions.shape[1]):
                        # find min action values 
                        if np.min(actions[:, i]) < METADATA[robot]['MIN_ACTION'][state_keys[i]]:
                            METADATA[robot]['MIN_ACTION'][state_keys[i]] = np.min(actions[:, i])
                        # find max action values 
                        if np.max(actions[:, i]) > METADATA[robot]['MAX_ACTION'][state_keys[i]]:
                            METADATA[robot]['MAX_ACTION'][state_keys[i]] = np.max(actions[:, i])

                except KeyError as e:
                    print(f"Key {e} is not in dataset {file}, and is therefore disregarded!")
                    continue

                # print('\t')
     
    with open('metadata.pkl', 'wb') as metadata: 
        print("Safe collected values as \'metadata.pkl\'")
        pkl.dump(METADATA, metadata)

    with open('metadata.pkl', 'rb') as metadata: 
        pkl.load(metadata)
        return metadata 

def main():
    #TODO: Add path from RoboNet dataset to config file
    #TODO: Change path to be machine-independent 
    path = '~ehrensberger/RoboNet'
    robots = ['sawyer', 'widowx', 'baxter', 'kuka', 'franka']
    files = [x for x in glob.glob(path+'*.hdf5') if any(robot in x for robot in robots)] 
    
    metadata = collect_metadata(files=files)
    print(metadata)

if __name__ == '__main__': 
    main()
