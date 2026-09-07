import os
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm 
from pathlib import Path
from termcolor import colored 
from typing import Dict, List, Optional, Tuple,  Union


def get_files(
    data_dir: Union[str, os.PathLike], 
    depth: bool=False, 
    robots: Optional[Union[str, List[str]]]=None, 
    tasks: Optional[Union[str, List[str]]]=None, 
    ) -> Tuple[List[str], List[str], List[str]]:
    
    data_dir = Path(data_dir)
    robots = [robots] if isinstance(robots, str) else list(robots) if robots is not None else None
    tasks = [tasks] if isinstance(tasks, str) else list(tasks) if tasks is not None else None
    
    all_files = [f for f in data_dir.iterdir() if f.is_file() and ("depth" in f.name) == depth]
    all_robots = list(set(f.stem.split("_")[-1] for f in all_files)) # no given robot -> all robots
    all_tasks = list(set(f.stem.split("_")[0] for f in all_files)) # no given task -> all tasks
    
    robots = _filtered_or_all(robots, all_robots)
    tasks  = _filtered_or_all(tasks, all_tasks)
    
    files = [
        str(f) for f in all_files
        if f.stem.split("_")[-1] in robots
        and f.stem.split("_")[0] in tasks
        ]
    
    print(colored(f"robots: {robots}", color="green"))
    print(colored(f"tasks: {tasks}", color="green"))
     
    return robots, tasks, files 

def _filtered_or_all(selected: Optional[List[str]], available: List[str]) ->  List[str]: 
    if not selected:
        return available

    filtered = [x for x in selected if x in available]
    return filtered if filtered else available

def get_depths(meta_dir: os.PathLike) -> pd.DataFrame: 
    depth_path = Path(meta_dir) / "depths.csv"
    return pd.read_csv(depth_path)  if depth_path.is_file() else pd.DataFrame()

def _get_gripper(meta_dir: Union[str, os.PathLike], files: List[Union[str, os.PathLike]], robots): 
    meta_dir = Path(meta_dir)
    meta_dir.mkdir(parents=True, exist_ok=True)
    meta_file = meta_dir / "meta.csv"
    
    df = pd.read_csv(meta_file) if meta_file.is_file() else pd.DataFrame()
    
    min = float("inf")
    max = float("-inf")
    
    new_columns = {}
    for file in tqdm(files, desc="Fetching gripper joint states from each hdf5 file"): 
        robot = Path(file).stem.split("_")[-1]
        
        if robot not in df.columns: 
            new_df = {}
            try: 
                with h5py.File(file, "r") as hf: 
                    demo_min = []
                    demo_max = []
                    data = hf["data"]
                    for demo in data.keys(): 
                        gripper_qpos = data[demo]["obs"]["robot0_gripper_qpos"][()]
                        
                        demo_min = np.min(gripper_qpos, axis=0)
                        demo_max = np.max(gripper_qpos, axis=0)
                       
                        min_mask = demo_min < df[robot]["min"]
                        max_mask = demo_max > df[robot]["max"]         
                
                        if min_mask.any(): 
                            new_df[robot]["min"][min_mask] = demo_min[min_mask]
                        if max_mask.any(): 
                            new_df[robot]["max"][max_mask] = demo_max[max_mask]
                    
            except Exception as e: 
                raise FileNotFoundError(colored(f"Could not open file: {file}", "red")) from e

        else: 
            continue
    
    return df 

def get_metadata(meta_dir: Union[str, os.PathLike], files: List[Union[str, os.PathLike]]) -> pd.DataFrame:
    meta_dir = Path(meta_dir)
    meta_dir.mkdir(parents=True, exist_ok=True)
    meta_file = meta_dir / "meta.csv"

    df = pd.read_csv(meta_file) if meta_file.is_file() else pd.DataFrame()

    new_columns = {}
    for file in tqdm(files, desc=colored("Fetching number of steps in each demo", "green"), colour="green"):
        file_str = str(file)
        
        if file_str not in df.columns:
            try:
                with h5py.File(file, "r") as hf:
                    data = hf["data"]
                    new_columns[file_str] = [demo_group["actions"].shape[0] for _, demo_group in data.items()]
            except (KeyError, OSError) as e:
                raise FileNotFoundError(colored(f"Could not open file: {file}", "red")) from e
                
    if new_columns:
        df = pd.concat([df, pd.DataFrame(new_columns)], axis=1)
        df.to_csv(meta_file, index=False)

    return df

def get_demo_list(metadata: pd.DataFrame, files: List[Union[str,os.PathLike]], window: int) -> Tuple[List[Tuple[str, str, int]], int]:     
    demo_map: List[Tuple[str, str, int]] = []
    min_horizon = float("inf")
    
    for file in tqdm(files, desc=colored("Building demo list", "green"), colour="green"):
        file_str = str(file)
        
        if file_str in metadata.columns:  
            valid_series = metadata[file_str].dropna()
            
            if len(valid_series) == 0: 
                continue
            else:
                n_steps_list = valid_series.astype(int).to_list()
                demos = [(file_str, f"demo_{idx}", n_steps) for idx, n_steps in zip(valid_series.index, n_steps_list)]
                demo_map.extend(demos)
                min_horizon = min(min_horizon, min(n_steps_list))

        else: 
            try: 
                with h5py.File(file, "r") as hf:
                    data = hf["data"]
                    for demo, demo_group in data.items():
                        n_steps = int(demo_group["actions"].shape[0])
                        demo_map.append((file_str, str(demo), n_steps)) 
                        min_horizon = min(min_horizon, n_steps)
            except Exception as e: 
                raise FileNotFoundError(colored(f"Could not open file: {file}", "red")) from e
                        
    if min_horizon < window: 
        print(f"The chosen size of the window is bigger than the smallest episode length! \n \
                Therefore, the size of the window gets changed from {window} to {int(min_horizon)}.")
        window = int(min_horizon)
    
    return demo_map, window

def get_demo_dict(metadata: pd.DataFrame, files: List[Union[str, os.PathLike]], window: int) -> Tuple[Dict[str, List[Tuple[str, str, int]]], int]: 
    demo_dict: Dict[str, List[Tuple[str, str, int]]] = {}
    min_horizon = float("inf")
    
    for file in tqdm(files, desc=colored("Building demo dict", "green"), colour="green"):
        file_str = str(file)
        key = Path(file).stem
        
        if file_str in metadata.columns:  
            valid_series = metadata[file_str].dropna()
            
            if len(valid_series) == 0: 
                demos = []
            else: 
                n_steps_list = valid_series.astype(int).to_list()
                demos = [(file_str, f"demo_{idx}", n_steps) for idx, n_steps in zip(valid_series.index, n_steps_list)]
                min_horizon = min(min_horizon, min(n_steps_list))
                
        else:
            demos = [] 
            try:  
                with h5py.File(file, "r") as hf:
                    data = hf["data"]
                    for demo, demo_group in data.items():
                        n_steps = int(demo_group["actions"].shape[0])
                        demos.append((file_str, str(demo), n_steps)) 
                        min_horizon = min(min_horizon, n_steps)        
            except Exception as e: 
                raise FileNotFoundError(colored(f"Could not open file: {file}", "red")) from e
                
        demo_dict[key] = demos
                   
    if min_horizon < window: 
        print(f"The chosen size of the window is bigger than the smallest episode length! \n \
            Therefore, the size of the window gets changed from {window} to {int(min_horizon)}.")
        window = int(min_horizon)
    
    return demo_dict, window

def get_depth_dict(
    files: Union[Union[str, os.PathLike], List[Union[str, os.PathLike]]], 
    meta_file: Optional[Union[str, os.PathLike]]=None
    ) -> pd.DataFrame:
    
    if isinstance(files, (str, os.PathLike)): 
        files = [Path(files)]
    else: 
        files = [Path(f) for f in files]
        
    if meta_file is not None: 
        meta_file = Path(meta_file)
        if not meta_file.exists(): 
            meta_file.parent.mkdir(parents=True, exist_ok=True)
    else: 
        meta_file = Path.cwd() / "depths.csv"
    
    if meta_file.exists(): 
        df = pd.read_csv(meta_file)
    else: 
        df = pd.DataFrame()
        
    depth_dict = {}
    keys = []
    
    for file in tqdm(files, desc=colored("Fetching minimal and maximal depth map values"), colour="green"): 
        key = file.stem
        if key in df.columns and df[key].notna().all(): 
                continue 
        else: 
            keys.append(key)
        
        if depth_dict.get(key, None) is None:
            depth_dict[key] = {}
        
        try: 
            with h5py.File(file, "r") as hf:    
                data = hf["data"]     
                min_dict = {}
                max_dict = {}   
                     
                for demo in hf.keys():
                    obs = data[demo]["obs"]
                    views = [v for v in obs.keys() if "depth" in v]
                    
                    for view in views: 
                        depth_map = obs[view][()]

                        if depth_dict[key].get(view, None) is None: 
                            depth_dict[key][view] = {}
                            depth_dict[key]["view"]["min"] = float("inf")
                            depth_dict[key]["view"]["max"] = float("-inf")
                        if min_dict[key].get(view, None) is None: 
                            min_dict[key][view] = []
                        if max_dict[key].get(view, None) is None: 
                            max_dict[key][view] = []
                        
                        min_dict[key][view].append(np.min(depth_map))
                        max_dict[key][view].append(np.max(depth_map))
                
                for view in min_dict[key].keys(): 
                    min_dict[key][view] = min(min_dict[key][view])    
                    max_dict[key][view] = max(max_dict[key][view])
                
                    if min(min_dict[key][view]) < depth_dict[key][view]["min"]: 
                        depth_dict[key][view]["min"] = min(min_dict[key][view])
                    
                    if max(min_dict[key][view]) > depth_dict[key][view]["max"]: 
                        depth_dict[key][view]["max"] = max(min_dict[key][view])
                 
        except Exception as e: 
            raise FileNotFoundError(colored(f"Could not open file: {file}", "red")) from e
    
    if len(keys) > 0: 
        df_new = pd.DataFrame(depth_dict, columns=key, index=[k for k in depth_dict[key][view].keys()])
        df = pd.concat([df, df_new], axis=1)
        df.to_csv(meta_file, index=True, encoding="utf-8")
    
    return df