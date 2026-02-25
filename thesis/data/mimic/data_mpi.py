import os 
import sys
import h5py 
import hydra
import numpy as np
from typing import List

from mpi4py import MPI 


comm = MPI.COMM_WORLD
rank = comm.Get_rank() 
size = comm.Get_size() 

def handle_file(file: os.PathLike, mode: str) -> np.float32: 
    n_actions = 0
        
    with h5py.File(file, mode) as hf:
        data = hf["data"]
        for demo in data: 
            actions = data[demo]["actions"][()]
            n_actions += actions.shape[0]      
    
    return n_actions 
        
def handle_files(files: List[os.PathLike]): 
    metric = 0
    for file in files: 
        metric += handle_file(file=file, mode="r")
        
    return metric 

@hydra.main(version_base = None, config_path="configs", config_name="data.yaml")
def main(cfg): 
    root = 0
    if rank == root: 
        file_dir = os.path.expanduser(cfg.file_dir)
        files = [os.path.join(file_dir, file) for file in os.listdir(file_dir) if ("depth" in file) == cfg.with_depth]
        n_files = len(files)
            
    else: 
        files = None 
        n_files = None
    
    files = comm.bcast(files, root=root)
    n_files = comm.bcast(n_files, root=root)
    
    if 0 < size < n_files:
        local_files = files[rank::size]
    elif size >= n_files: # more processes than files or as many processes as files 
        if rank < n_files:
            local_files = [files[rank]] # list 
        else: 
            
            MPI.Finalize() 
            sys.exit() 
    else: 
        raise ValueError("No hdf5 files in the given dataset!")
        
    local_metric = handle_files(local_files)   
    reduced_metric = comm.reduce(local_metric, op=MPI.SUM, root=0)
    
    if rank == 0: 
        print("root") 
        print(reduced_metric)
        

if __name__ == "__main__": 
    main() 