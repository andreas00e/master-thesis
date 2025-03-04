import os
import glob
import copy  
import h5py
import tqdm 
import random
import numpy as np
import pickle as pkl 
 
from torch.utils.data import Dataset, DataLoader

import time 
from functools import wraps

from dataclasses import dataclass, field 

from robonet.datasets.util.metadata_helper import load_metadata
from robonet.datasets.util.hdf5_loader import *

# TODO: Add description from where this was copied 
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
    
    # Check if metadata has already been saved 
    if os.path.isfile('metadata.pkl'): 
        print("metadata.pkl has already been created!") 
        return None
    
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
                    # Current hdf5 file does not contain keys qpos and/or qvel!
                    continue

                # print('\t')
     
    with open('metadata.pkl', 'wb') as metadata: 
        print("Safe collected values as \'metadata.pkl\'")
        pkl.dump(METADATA, metadata)

class RoboNetDataset(Dataset):
    def __init__(self, path, robots, horizon):
        super().__init__()

        self.path = path 
        self.robots = robots 
        self.horizon = horizon

        self.hparams = HParams(action_mismatch=ACTION_MISMATCH.CLEAVE)
        self.files = [x for x in glob.glob(self.path +'*.hdf5') if any(robot in x for robot in self.robots)]
        # Sort hdf5 files to be in ascending order 
        # self.files = sorted(self.files, key=lambda x: int(x.split('.')[-2].split('_')[-1].replace('traj', '')))
         
        if not os.path.isfile('./metadata.pkl'):
            collect_metadata(self.files)

        with open('./metadata.pkl', 'rb') as metadata: 
            self.metadata = pkl.load(metadata)

        data_folder = '/'.join(self.path.split('/')[:-1])
        self.file_metadata = load_metadata(data_folder)
        self.data = []

    def __len__(self): 
        return len(self.files)
    
    def __getitem__(self, index):
        self.data.clear()
        # TODO: Try to get rid of the try except statements 
        try: 
            with h5py.File(self.files[index], 'r') as hf:
                try:
                    # load images, actions, and states with robonet functions 
                    imgs, actions, states = load_data(f_name=self.files[index], file_metadata=self.file_metadata.get_file_metadata(self.files[index]), hparams=self.hparams)
                    qpos = np.array(hf['env']['qpos'])
                    qvel = np.array(hf['env']['qpos'])
                    robot = hf['metadata'].attrs['robot']

                    # load embodiment specific max and min values 
                    metadata =  self.metadata[robot.upper()]
                    min_state_xyz = np.min(list(metadata['MIN_STATE'].values())[:3])
                    max_state_xyz = np.max(list(metadata['MAX_STATE'].values())[:3])
                    min_action_xyz = np.min(list(metadata['MIN_ACTION'].values())[:3])
                    max_action_xyz = np.max(list(metadata['MAX_ACTION'].values())[:3])
                    
                    episode = dict()                        
                    for i in range(states.shape[0]): 
                        # add normalized state to dict 
                        state_xyz = (states[i, :3]-min_state_xyz)/(max_state_xyz-min_state_xyz)
                        state_yaw = np.array([(states[i, 3]-metadata['MIN_STATE']['YAW'])/(metadata['MAX_STATE']['YAW']-metadata['MIN_STATE']['YAW'])])
                        state_gripper = np.array([(states[i, 4]-metadata['MIN_STATE']['GRIPPER'])/(metadata['MAX_STATE']['GRIPPER']-metadata['MIN_STATE']['GRIPPER'])])
                        episode['state'] = np.expand_dims(np.concatenate((state_xyz, state_yaw, state_gripper)), axis=1) # -> np.darray(5, 1)
                        # print(f"episode['state'] has shape: {episode['state'].shape}")
                        
                        # add normalized actions to dict 
                        actions = np.array([actions[i+j, :] if i+j < actions.shape[0] else np.zeros((actions.shape[1],)) for j in range(self.horizon)])
                        actions_xyz = np.where(actions[:, :3] != 0, (actions[:, :3]-min_action_xyz)/(max_action_xyz-min_action_xyz), 0)
                        actions_yaw = np.expand_dims(np.where(actions[:, 3] != 0, 
                            (actions[:, 3]-metadata['MIN_ACTION']['YAW'])/(metadata['MAX_ACTION']['YAW']-metadata['MIN_ACTION']['YAW']), 0),axis=1)
                        
                        if actions.shape[1] == 5: 
                            actions_gripper = np.expand_dims(np.where(actions[:, 4] != 0, 
                                (actions[:, 4]-metadata['MIN_ACTION']['GRIPPER'])/(metadata['MAX_ACTION']['GRIPPER']-metadata['MIN_ACTION']['GRIPPER']), 0), axis=1)
                        else: 
                            actions_gripper = np.zeros((self.horizon, 1))

                        episode['actions'] = np.concatenate((actions_xyz, actions_yaw, actions_gripper), axis=1)
                        # print(f"episode['action'] has shape: {episode['actions'].shape}")

                        # TODO: check how position of first camera is described inpaper #TODO: check how to adequately normalize images
                        episode['image'] = imgs.squeeze()[i, ...]

                        # add normalized joint position and velocity to dict 
                        episode['qpos'] = np.expand_dims((qpos[i, :]-metadata['MIN_Q_POS'])/(metadata['MAX_Q_POS']-metadata['MIN_Q_POS']), axis=1) # normalized joint position
                        episode['qvel'] = np.expand_dims((qvel[i, :]-metadata['MIN_Q_VEL'])/(metadata['MAX_Q_VEL']-metadata['MIN_Q_VEL']), axis=1)# normalized joint velocity

                        # add robot name to dict 
                        episode['robot'] = robot # current robot 

                        self.data.append(copy.deepcopy(episode))
                        episode.clear()

                except KeyError as e:  
                    # print(f"KeyError with problem-causing key: {e}")
                    # if HDF5 file does not contain qpos or qvel return a different element at random 
                    rand_index = random.randint(0, self.__len__()-1)
                    return self.__getitem__(rand_index)
                
        except FileNotFoundError as e:
            # print("HDfF5 file {e} could not be found!")
            return None 

        return self.data

def main(): 
    path = '~/ehrensberger/RoboNet/hdf5/' 
    path = os.path.expanduser(path)

    # list of all robots with qpos and qvel data 
    robots = ['sawyer', 'widowx', 'baxter', 'kuka', 'franka']
    horizon = 8 

    roboNetDataset = RoboNetDataset(path=path, robots=robots, horizon=horizon)
    roboNetDataLoader = DataLoader(dataset=roboNetDataset, batch_size=1, num_workers=4)

    for i in tqdm.tqdm(roboNetDataLoader): 
        a = i 

if __name__ == '__main__': 
    main()