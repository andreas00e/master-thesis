import os
import glob
import copy  
import h5py
import tqdm 
import numpy as np
import pickle as pkl 

import torch 
from torch.utils.data import Dataset

import time 
from functools import wraps


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

@time_it
def collect_metadata(path):
    state_keys = list(STATE.keys())
    
    # Check if metadata has already been saved 
    try: 
        with open('metadata.pkl', 'rb') as metadata: 
            metadata = pkl.load(metadata)
            return 0 

    except FileNotFoundError: 
        print("metadata.pkl file has not been created yet!")

    for file in tqdm.tqdm(glob.glob(path+'*.hdf5'), desc="Finding min and max values for normalization", colour='green'): 
        with h5py.File(file, 'r') as hf: 
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
                                    
                    actions = np.array(hf['policy']['actions'])
                    # print(actions.shape)
                    for i in range(actions.shape[1]): 
                        # find min action values 
                        if np.min(actions[:, i]) < METADATA[robot]['MIN_ACTION'][state_keys[i]]:
                            METADATA[robot]['MIN_ACTION'][state_keys[i]] = np.min(actions[:, i])
                        # find max action values 
                        if np.max(actions[:, i]) > METADATA[robot]['MAX_ACTION'][state_keys[i]]:
                            METADATA[robot]['MAX_ACTION'][state_keys[i]] = np.max(actions[:, i])

                except KeyError: 
                    # print("Current hdf5 file does not contain keys qpos and/or qvel!")
                    continue

                # print('\t')
     
    with open('metadata.pkl', 'wb') as metadata: 
        print("Safe collected values as \'metadata.pkl\'")
        pkl.dump(METADATA, metadata)

class RoboNetDataset(Dataset):
    def __init__(self, robots=['sawyer', 'widowx', 'baxter', 'kuka', 'franka'], horizon=16):
        super().__init__()
        path = '~/ehrensberger/RoboNet/hdf5/' 
        path = os.path.expanduser(path)
        files = [x for x in glob.glob(path+'*.hdf5') if any(robot in x for robot in robots)]
        
        self.horizon = horizon
        self.data = []
        
        for i in files:
            try: 
                buf = open(i, 'rb').read()
                with h5py.File(i, 'r') as hf:
                    try: 
                        # print(hf['metadata'].attrs['low_bound'])
                        # print(hf['metadata'].attrs['high_bound'])

                        # print(np.array(hf['env']['high_bound']))

                        # print(hf['metadata'].attrs.keys())
                        for k in hf['metadata'].attrs.keys():
                            print(hf['metadata'].attrs[k])
                            # print(hf['metadata']['object_classes'])
                        break
                        episode = dict()
                        # n_rows = np.array(print(f['metadata'])f['policy']['actions'].shape[0])
                        
                        j = 0 
                        while(j+self.horizon-1 < n_rows):
                            
                            break
                            # episode['state'] = np.array(f['env']['state'])[j, :]
                            #episode['qpos'] = np.array(f['env']['qpos'])[j, :]
                            #episode['qvel'] = np.array(f['env']['qvel'])[j, :]
                            #print(f"{i}: {episode['state'][-1]}")
                            #episode['actions'] = np.array(f['policy']['actions'])[j:j+horizon-1, :]
                            #self.data.append(episode)
                            #episode.clear()
                            j+=1
                    except KeyError:
                        
                        qpos = None
                        qvel = None 
                                
            except FileNotFoundError: 
                print("HDfF5 file could not be found!")
        
    def __len__(self): 
        return 5 
    
    def __getitem__(self, index):
        return super().__getitem__(index)

def main(): 
    path = '~/ehrensberger/RoboNet/hdf5/' 
    path = os.path.expanduser(path)

    # RoboNetDataset()
    collect_metadata(path)
    with open('metadata.pkl', 'rb') as metadata: 
        metadata = pkl.load(metadata)
        print(metadata)

if __name__ == '__main__': 
    main()