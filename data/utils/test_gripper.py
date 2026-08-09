import numpy as np
import pandas as pd 

file_dir = "/home/bing_TUM/ehrensberger/master-thesis/thesis/data/metadata/gripper_state.csv"
new_file_dir = "/home/bing_TUM/ehrensberger/master-thesis/thesis/data/metadata/gripper_state_robot.csv"

df = pd.read_csv(file_dir)

new = {}
for _, row in df.iterrows(): 
    robot = row["robot"]
    task = row["task"]
    
    key = f"{robot}_{task}"
    values = row.to_numpy()[2:].astype(float)    
    new[key] = values

keys = list(new.keys()) 

final = {}
for i, key in enumerate(keys): 
    if i % 2 != 0: 
        continue 
    
    one_min = new[keys[i]][0::2]    
    two_min = new[keys[i+1]][0::2]
    
    one_max = new[keys[i]][1::2]
    two_max = new[keys[i+1]][1::2]
    
    min_mask = one_min < two_min
    max_mask = one_max > two_max
    
    two_min = two_min.copy() 
    two_max = two_max.copy() 
    
    two_min[min_mask] = one_min[min_mask]
    two_max[max_mask] = one_max[max_mask]
    
    robot = key.split("_")[0]
    final[f"{robot}_min"] = two_min
    final[f"{robot}_max"] = two_max

final = pd.DataFrame(final)
final.to_csv(new_file_dir, index=False)