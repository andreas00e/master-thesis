import os 
import sys
import h5py 
import hydra
import numpy as np
from collections import defaultdict
from typing import Callable, List, Tuple, Union


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

def get_depth_limits(file: os.PathLike) -> int: 
    depth_limits = {}
    
    with h5py.File(file, "r+") as hf: 
        data = hf["data"] 
        
        for demo in data: 
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
    
    return depth_limits 
          
    #     if hf.get("depth_limits", None):
    #         hf["depth_limits"][()] = depth_limits 
    #     else: 
    #         hf.create_dataset(name="depth_limits", data=depth_limits)
                    
    # return 1           

def get_n_actions(file: os.PathLike, mode: str) -> np.float32: 
    n_actions = 0
        
    with h5py.File(file, mode) as hf:
        data = hf["data"]
        for demo in data: 
            actions = data[demo]["actions"][()]
            n_actions += actions.shape[0]      
    
    return n_actions 
        
def handle_files(function: Callable[[str], Union[int, np.float32]], files: List[os.PathLike]): 
    results = []
    
    for file in files:
        result = FUNCTIONS[function](file=file)
        results.extend(result)
        
    return results 


FUNCTIONS = {
    "anotate_files": anotate_files, 
    "get_depth_limits": get_depth_limits, 
    "get_n_actions": get_n_actions
    }


@hydra.main(version_base = None, config_path="../../configs", config_name="data.yaml")
def main(cfg): 
    root = 0
    
    if rank == root: 
        file_dir = os.path.expanduser(cfg.file_dir)
        files = [os.path.join(file_dir, file) for file in os.listdir(file_dir) if ("depth" in file) == cfg.with_depth]
        print(f"Performing pre-processing operations on: {files}")
        n_files = len(files)
    else: 
        files = None 
        n_files = None
    
    files = comm.bcast(files, root=root)
    n_files = comm.bcast(n_files, root=root)
    
    if 0 < size < n_files: # less processes as files
        local_files = files[rank::size]
    elif size >= n_files: # more processes than files or as many processes as files 
        if rank < n_files:
            local_files = [files[rank]] # list 
        else: 
            MPI.Finalize() 
            sys.exit() 
    else: 
        raise ValueError("No hdf5 files in the given dataset!")
    
    functions: list = cfg.functions 
    for function in functions: 
        local_metric = handle_files(function=function, files=local_files)   
    
    final_metric = comm.gather(local_metric, root=0)
    if rank == 0: 
        print(final_metric)
    
    # reduced_metric = comm.reduce(local_metric, op=MPI.SUM, root=0)
    
    # if rank == 0: 
    #     print("root") 
    #     print(reduced_metric)
        

if __name__ == "__main__": 
    main() 