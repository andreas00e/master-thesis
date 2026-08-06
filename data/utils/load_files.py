import os
import h5py
import numpy as np 
import pandas as pd

from typing import  List, Optional, Tuple, Union

  
def get_files(
    data_dir: os.PathLike, 
    depth: str, 
    robots: Optional[Union[str, List]] , 
    tasks: Optional[Union[str, List]]) -> List[os.PathLike]:
    
    all_files = [file for file in os.listdir(data_dir) if ("depth" in file) == depth]
    all_robots = list(set(f.split(".")[-2].split("_")[-1] for f in all_files)) # no given robot -> all robots
    all_tasks = list(set(f.split(".")[-2].split("_")[0] for f in all_files)) # no given task -> all tasks
    
    robots = _filtered_or_all(robots, all_robots)
    tasks  = _filtered_or_all(tasks, all_tasks)
    files = [os.path.join(data_dir, file) for file in all_files
        if any(robot in file for robot in robots)
        and any(task in file for task in tasks)]
    
    return files 

def _filtered_or_all(selected: Union[str, List], available: List[str]) -> List[str]: 
    if selected is None:
        return available
    filtered = [x for x in selected if x in available]
    return filtered if filtered else available

def get_depths(meta_dir: os.PathLike) -> pd.DataFrame: 
    depth_path = os.path.join(meta_dir, "depths.csv")
    if os.path.isfile(depth_path):
        df = pd.read_csv(depth_path)
    else: 
        df = pd.DataFrame()
    return df

def get_metadata(meta_dir: os.PathLike, files: List[os.PathLike]) -> pd.DataFrame: 
    meta_path = os.path.join(meta_dir, "meta.csv")
    if os.path.isfile(meta_path):
        df = pd.read_csv(meta_path)
    else:
        df = pd.DataFrame()

    for file in files:
        if file in df.columns:
            continue

        with h5py.File(file, "r") as hf:
            data = hf["data"]
            demos = [data[demo]["actions"][()].shape[0] for demo in data.keys()]
        df[file] = demos
    df.to_csv(meta_path, index=False)
    return df 

def get_demomap(files, window): # -> List[List[os.PathLike, int, int, None]]: 
    H = np.inf # H: min episode duration -> max possible window size
    demo_map = []

    for file in files:
        with h5py.File(file, "r") as hf:
            for demo in hf["data"].keys():
                n_steps = hf["data"][demo]["actions"][()].shape[0]
                
                if n_steps < H: # number of steps in the demo is smaller than the horizon 
                    H = n_steps # horizon is now equal to the number of steps 

                demo_map.append([file, demo, n_steps]) 
                         
    if H < window: 
        print(f"The chosen size of the window is bigger than the smallest episode length! \n \
                Therefore, the size of the window gets changed from {window} to {H}.")
        window = H
    
    return demo_map