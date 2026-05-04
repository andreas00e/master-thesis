import os 
import glob
import h5py

import numpy as np 

DEMOS = {}


def final_bin(position: np.array) -> str:
    #  Offsets and boundaries found from inspecting 
    # robosuite/robosuite/models/assets/arenas/bins_arena.xml 
    x_off, y_off = 0.1, 0.28
    x, y = position[0], position[1]
    
    # Object is in box 1 at end of trajectory/segment
    if -0.2 < x-x_off < 0 and -0.25 < y-y_off < 0: 
        return 1 
    # Object is in box 2 at end of trajectory/segment 
    elif -0.2 < x-x_off < 0 and 0 < y-y_off < 0.25: 
        return 2
    # Object is in box 3 at end of trajectory/segment
    elif 0 < x-x_off < 0.2 and -0.25 < y-y_off < 0: 
        return 3
    # Object is in box 4 at end of trajectory/segment
    elif 0 < x-x_off < 0.2 and 0 < y-y_off < 0.25: 
        return 4
    else: 
        raise ValueError("Given object position lies outside of the four possible bins")


def main(): 
    load_dir = '~/ehrensberger/mimicgen/datasets/core'
    load_dir = os.path.expanduser(load_dir)
    
    tasks = ['pick_place']
    files = [glob.glob(f'{os.path.join(load_dir, task)}*hdf5')[0] for task in tasks]
    
    for file in files: 
        with h5py.File(file, 'r') as hf:
            counter=0
            for demo in hf['data'].keys():
                # print(hf['data'][demo]['actions'].shape)
                episode = {'bread': [0, 0], 'can': [0, 0], 'cereal': [0, 0], 'milk': [0, 0]} 
                objects = hf['data'][demo]['datagen_info']['object_poses'].keys()
                # print(f"Current demo: {demo}"
                # Iterate over all available objetcts
                for obj in objects:
                    # x-position of current object for all time steps of trajectory
                    object_pose = hf['data'][demo]['datagen_info']['object_poses'][obj][:, 0, 3]
                    object_pose = np.round(object_pose, 3)
                    # object_pose = int(object_pose*1000)/1000
                    
                    segment_start = np.argwhere(object_pose==object_pose[0]).squeeze().max()
                    segment_end = np.argwhere(object_pose==object_pose[-1]).squeeze().min()
                    
                    print(segment_start)
                    print(segment_end)
                    
                    episode[obj] = [segment_start, segment_end]
                

                    
                episode = dict(sorted(episode.items(), key=lambda x: x[1][0]))
                
                for i, key in enumerate(episode.keys()):
                    if i == 0: 
                        episode[key][0] = 0 
                    elif i == len(episode)-1: 
                        episode[key][1] = hf['data'][demo]['actions'].shape[0]
                        break
                    episode_list = list(episode)
                    current_key = episode_list[i]
                    next_key = episode_list[i+1]
                    
                    difference = episode[next_key][0]-episode[current_key][1]
                    
                    if difference%2 == 0: 
                        episode[current_key][1] += int(difference/2)
                        episode[next_key][0] -= int(difference/2)-1
                    elif difference%2 == 1: 
                        episode[current_key][1] += int(difference/2)
                        episode[next_key][0] -= int(difference/2)
                    

                print(episode)
                if list(episode)[0] != 'milk': 
                    counter+=1
                
                # print(episode.items())

                    
                    
                    
                    
                    
                    # X-position of current object for all time steps of trajectory
                #     object_pose = hf['data'][demo]['datagen_info']['object_poses'][object][:, :3, 3]
                #     object_pose = np.round(object_pose, 3)
                    
                #     object_final_bin = final_bin(object_pose[-1, :2])
                #     print(f"At the end of the tjractory object {object} is in bin {object_final_bin}")
                    
                # print("-------------------------------------------------------------------------")
                    
                # if demo=='demo_50': 
                #     exit()
        print(counter)


if __name__ == '__main__': 
    main()