import os
import h5py
import numpy as np 
import pandas as pd
from tqdm import tqdm 
from termcolor import colored 


from typing import List, Optional, Union

  
def get_files(
    data_dir: os.PathLike, 
    depth: Optional[bool]=None, 
    robots: Optional[Union[str, List]]=None, 
    tasks: Optional[Union[str, List]]=None, 
    ) -> List[os.PathLike]:
    
    all_files = [file for file in os.listdir(data_dir) if ("depth" in file) == depth]
    all_robots = list(set(f.split(".")[-2].split("_")[-1] for f in all_files)) # no given robot -> all robots
    all_tasks = list(set(f.split(".")[-2].split("_")[0] for f in all_files)) # no given task -> all tasks
    
    robots = _filtered_or_all(robots, all_robots)
    tasks  = _filtered_or_all(tasks, all_tasks)
    files = [
        os.path.join(data_dir, file) for file in all_files
        if any(robot in file for robot in robots)
        and any(task in file for task in tasks)
        ]
    
    # if stage == "test": 
    #     random.shuffle(files)
    #     files = [
    #         next((f for f in all_files if robot in f and task in f), None)
    #         for robot in robots
    #         for task in tasks
    #     ]
     
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
    if not os.path.isdir(meta_dir):
        os.mkdir(meta_dir)
        
    meta_file = os.path.join(meta_dir, "meta.csv")
    df = pd.read_csv(meta_file) if os.path.isfile(meta_file) else pd.DataFrame()

    new_cols = {}
    for file in tqdm(files, desc=colored("Fetching length of each demo", "green"), colour="green"):
        if file not in df.columns:
            with h5py.File(file, "r") as hf:
                data = hf["data"]
                new_cols[file] = [demo_group["actions"].shape[0] for _, demo_group in data.items()]

    if new_cols:
        df = pd.concat([df, pd.DataFrame(new_cols)], axis=1)
        df.to_csv(meta_file, index=False)

    return df

def get_demomap(meta_data: pd.DataFrame, files: List[os.PathLike], window: int): 
    df = meta_data
    
    demo_map = []
    min_horizon = np.inf
    
    for file in tqdm(files, desc=colored("Fetching mapping from files to individual demos", "green"), colour="green"): 
        if file in df.columns:  
            f = [file] * len(df)
            idx = [f"demo_{idx}" for idx in df.index]
            n_steps = list(df[file].values)
            
            demos = list(zip(f, idx, n_steps))
            demo_map.extend(demos)

        else: 
            with h5py.File(file, "r") as hf:
                data = hf["data"]
                
            for demo, demo_group in data.items():
                n_steps = demo_group["actions"].shape[0]
                demo_map.append([file, demo, n_steps]) 
                min_horizon = min(min_horizon, n_steps)
                        
    if min_horizon < window: 
        print(f"The chosen size of the window is bigger than the smallest episode length! \n \
                Therefore, the size of the window gets changed from {window} to {min_horizon}.")
        window = int(min_horizon)
    
    return demo_map, window