import os 
import h5py
import glob 
import argparse
import numpy as np
import multiprocessing

import time 
from functools import wraps 

from robonet.datasets.util.metadata_helper import load_metadata
from robonet.datasets.util.hdf5_loader import *

from utils.utils import collect_metadata, HParams

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

class hdf5_Creator(): 
    def __init__(self, load_dir, save_dir, robot, horizon): 
        super().__init__()
        self.load_path = load_dir  
        self.save_path = save_dir
        self.robot = robot 
        # TODO: Check if this is really necessary here or only in the dataloader
        self.horizon = horizon 

        self.files = [x for x in glob.glob(self.load_path+'*.hdf5') if robot in x]
        self.metadata = collect_metadata(self.files)
        
        # adopted functionality from /robonet/datasets/util/ 
        self.hparams = HParams(action_mismatch=ACTION_MISMATCH.CLEAVE)
        data_folder = '/'.join(self.load_path.split('/')[:-1])
        self.file_metadata = load_metadata(data_folder)
    
    def create_hdf5(self, i, file): 
        try: 
            with h5py.File(file, 'r') as hf_load:
                try:
                    qpos = np.array(hf_load['env']['qpos'])
                    qvel = np.array(hf_load['env']['qpos'])
                    robot = hf_load['metadata'].attrs['robot']
                    print(f"Name of robot before operations: {robot}")

                    # load images, actions, and states with robonet functions 
                    imgs, actions, states = load_data(f_name=file, file_metadata=self.file_metadata.get_file_metadata(file), hparams=self.hparams)

                    # load embodiment specific max and min values 
                    metadata =  self.metadata[robot.upper()]
                    min_state_xyz = np.min(list(metadata['MIN_STATE'].values())[:3])
                    max_state_xyz = np.max(list(metadata['MAX_STATE'].values())[:3])
                    min_action_xyz = np.min(list(metadata['MIN_ACTION'].values())[:3])
                    max_action_xyz = np.max(list(metadata['MAX_ACTION'].values())[:3])
                                    
                    for j in range(states.shape[0]): 
                        hdf5 = robot+'_traj_{}_{}.hdf5'.format(i, j)
                        with h5py.File(os.path.join(self.save_path, hdf5), 'w') as hf_write: 
                            # add normalized state to dict 
                            state_xyz = (states[j, :3]-min_state_xyz)/(max_state_xyz-min_state_xyz)
                            state_yaw = np.array([(states[j, 3]-metadata['MIN_STATE']['YAW'])/(metadata['MAX_STATE']['YAW']-metadata['MIN_STATE']['YAW'])])
                            state_gripper = np.array([(states[j, 4]-metadata['MIN_STATE']['GRIPPER'])/(metadata['MAX_STATE']['GRIPPER']-metadata['MIN_STATE']['GRIPPER'])])
                            hf_write['state'] = np.expand_dims(np.concatenate((state_xyz, state_yaw, state_gripper)), axis=1)
                            # print(f"episode['state'] has shape: {episode['state'].shape}")
                            
                            # add normalized actions to dict 
                            actions = np.array([actions[j+k, :] if j+k < actions.shape[0] else np.zeros((actions.shape[1],)) for k in range(self.horizon)])
                            actions_xyz = np.where(actions[:, :3] != 0, (actions[:, :3]-min_action_xyz)/(max_action_xyz-min_action_xyz), 0)
                            actions_yaw = np.expand_dims(np.where(actions[:, 3] != 0, 
                                (actions[:, 3]-metadata['MIN_ACTION']['YAW'])/(metadata['MAX_ACTION']['YAW']-metadata['MIN_ACTION']['YAW']), 0),axis=1)
                            
                            if actions.shape[1] == 5: 
                                actions_gripper = np.expand_dims(np.where(actions[:, 4] != 0, 
                                    (actions[:, 4]-metadata['MIN_ACTION']['GRIPPER'])/(metadata['MAX_ACTION']['GRIPPER']-metadata['MIN_ACTION']['GRIPPER']), 0), axis=1)
                            else: 
                                actions_gripper = np.zeros((self.horizon, 1))

                            hf_write['actions'] = np.concatenate((actions_xyz, actions_yaw, actions_gripper), axis=1)
                            # print(f"episode['action'] has shape: {episode['actions'].shape}")

                            # TODO: check how position of first camera is described inpaper #TODO: check how to adequately normalize images
                            hf_write['image'] = imgs.squeeze()[j, ...]

                            # add normalized joint position and velocity to dict 
                            hf_write['qpos'] = np.expand_dims((qpos[j, :]-metadata['MIN_Q_POS'])/(metadata['MAX_Q_POS']-metadata['MIN_Q_POS']), axis=1) # normalized joint position
                            hf_write['qvel'] = np.expand_dims((qvel[j, :]-metadata['MIN_Q_VEL'])/(metadata['MAX_Q_VEL']-metadata['MIN_Q_VEL']), axis=1) # normalized joint velocity

                            # add robot name to dict 
                            hf_write['robot'] = robot # current robot
                            print(f"Name of robot after operations: {robot}")
                            print(f"Type of robot after operations: {type(robot)}")
                            exit()

                except KeyError as e:  
                    # print(f"KeyError with error message: {e}")
                    return

        except FileNotFoundError as e:
            # print(f"FileNotFoundError with error message {e}")
            return 
    
    @time_it
    def create_dataset(self):
        indices = [i for i in range(len(self.files))]  
        with multiprocessing.Pool(multiprocessing.cpu_count()) as pool: 
            pool.starmap(self.create_hdf5, zip(indices, self.files))

    @property
    def _len(self): 
        return len(self.files)
    
    @property 
    def _metadata(self): 
        return self.metadata[self.robot.upper()] 

def main(): 
    # TODO: Think about how to change/incorporate horizon more 'organically'
    parser = argparse.ArgumentParser(description='specify source and destination directory of dataset, robot (, and horizon(s))')
    parser.add_argument('--load_dir', type=str, help='source directory of RoboNet hdf5 files')
    parser.add_argument('--save_dir', type=str, help='destination direcoty of customized RoboNet hdf5 files')
    parser.add_argument('--robot', type=str, help='robot which hdf5 files are to be customized')
    parser.add_argument('--horizon', type=int, help="action lookout of network input")
    args = parser.parse_args()

    args.load_dir = os.path.expanduser(args.load_dir)
    args.save_dir = os.path.expanduser(args.save_dir)

    hdf5_creator = hdf5_Creator(**vars(args))
    hdf5_creator.create_dataset()
    
if __name__ == '__main__': 
    main()