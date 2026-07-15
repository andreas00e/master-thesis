import os 
import sys
import h5py
import hydra
import numpy as np
from tqdm import tqdm 
from collections import defaultdict
from typing import Callable, Dict, List, Tuple, Union

from mpi4py import MPI 

comm = MPI.COMM_WORLD
rank = comm.Get_rank() 
size = comm.Get_size()


def anotate_files(file: os.PathLike) -> int: 
    n_steps = {}
    
    with h5py.File(file , "r+") as hf: 
        data = hf["data"]
        
        for demo in data: 
            n_steps[demo] = data[demo]["actions"][()].shape[0]
          
        n_steps = np.array(n_steps, np.float32)  
        if hf.get("n_steps", None): 
            hf["n_steps"][()] = n_steps
        else: 
            hf.create_dataset(name="n_steps", data=n_steps)
    
    return 1 

def get_depth_limits(file: os.PathLike) -> List[Tuple[str, Dict[str, np.float32]]]: 
    result = None 
    depth_limits = {}
    
    with h5py.File(file, "r") as hf: 
        data = hf["data"] 
        
        for demo in tqdm(data): 
            for member in data[demo]["obs"]: 
                if "depth" in member: 
                    depth_data = data[demo]["obs"][member][()]
                    
                    if depth_limits.get(f"{member}_min", None) is None: 
                        depth_limits[f"{member}_min"] = [] 
                    if depth_limits.get(f"{member}_max", None) is None: 
                        depth_limits[f"{member}_max"] = []                         

                    depth_limits[f"{member}_min"].append(np.min(depth_data))
                    depth_limits[f"{member}_max"].append(np.max(depth_data))
        
        for key in depth_limits: 
            if "min" in key: 
                depth_limits[key] = np.min(depth_limits[key])
            elif "max" in key: 
                depth_limits[key] = np.max(depth_limits[key])
                
    result = [file, depth_limits]        
    return result           

def get_n_actions(file: os.PathLike, mode: str) -> np.float32: 
    n_actions = 0
        
    with h5py.File(file, mode) as hf:
        data = hf["data"]
        for demo in data: 
            actions = data[demo]["actions"][()]
            n_actions += actions.shape[0]      
    
    return n_actions 
        
def handle_files(function: Callable[[str], Union[int, np.float32]], files: List[os.PathLike]): 
    results = {}
    
    for file in files:
        result = FUNCTIONS[function](file=file)
        results[file] = result
        
    return results 

FUNCTIONS = {
    "anotate_files": anotate_files, 
    "get_depth_limits": get_depth_limits, 
    "get_n_actions": get_n_actions
    }


@hydra.main(config_path="../../cfgs", config_name="data.yaml", version_base=None)
def main(cfg): 
    root = 0
    
    if rank == root: 
        file_dir = os.path.expanduser(cfg.file_dir)
        files = [os.path.join(file_dir, file) for file in os.listdir(file_dir) if ("depth" in file) == cfg.with_depth]
        n_files = len(files)
        final_dict = (zip(files, [defaultdict]*n_files))
        print("LET'S GET THIS PASSING STARTED!")
    else: 
        files = None 
        n_files = None
        final_dict = None 
    
    files = comm.bcast(files, root=root)
    n_files = comm.bcast(n_files, root=root)
    final_dict = comm.bcast(final_dict, root=root)
    
    if 0 < size < n_files: # less processes than files
        local_files = files[rank::size]
    elif size >= n_files: # more processes than files or as many processes as files 
        if rank < n_files:
            local_files = [files[rank]] # list 
        else: 
            MPI.Finalize() 
            sys.exit() 
    else: 
        raise FileNotFoundError
    
    functions: list = cfg.functions 
    for function_name in functions: 
        if rank == 0: 
            final_metric = 0 
            metric_name = function_name.replace("get_", "")
        else: 
            metric_name = None
            
        local_metrics = handle_files(function=function_name, files=local_files)   
        for file in local_files: 
            final_dict[file][metric_name] = local_metrics[file]
        
        final_metric = comm.gather(local_metrics, root=0)
        if rank == 0: 
            print(final_metric)
        
if __name__ == "__main__": 
    main() 