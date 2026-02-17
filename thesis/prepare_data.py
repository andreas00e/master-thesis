import os 
import cv2
import h5py
import numpy as np
from tqdm import tqdm 

                            
def prepare_hdf5(file_dir: os.PathLike) -> None:
    for file in tqdm(iterable=os.listdir(file_dir), colour="green"): 
        with h5py.File(os.path.join(file_dir, file) , "r+") as hf: 
            n_steps = []
            data = hf["data"]

            for demo in data: # group
                n_steps_demo = data[demo]["actions"][()].shape[0]
                data[demo]["n_steps"][()] = n_steps_demo
                n_steps.append((np.string_(demo),  n_steps_demo))
            
            if hf.get("n_steps", None): # file
                hf["n_steps"][()] = n_steps 
            else: 
                hf.create_dataset(name="n_steps", data=np.array(n_steps))          


def get_depth_limits(file: os.PathLike):
    if "depth" not in file:
        return
    
    with h5py.File(file, "r+") as hf:
        if hf.get("depth_limits") is not None:
            return
        
        min_depth_values, max_depth_values = [], []
        for demo in hf["data"]:
            for member in hf["data"][demo]["obs"]:
                if "depth" in member:
                    depth_data = hf["data"][demo]["obs"][member][()]
                    min_depth_values.append(np.min(depth_data))
                    max_depth_values.append(np.max(depth_data))
        
        if min_depth_values and max_depth_values:
            min_depth = np.min(min_depth_values)
            max_depth = np.max(max_depth_values)
            
            depth_limits = np.array([min_depth, max_depth], dtype=np.float32)
            
            if hf.get("depth_limits") is not None: 
                hf["depth_limits"][()] = depth_limits
            else: 
                hf.create_dataset(name="depth_limits", data=depth_limits)                    
 
           
def main(): 
    file_dir = "~/ehrensberger/master-thesis/mimicgen/datasets/robot"
    file_dir = os.path.expanduser(file_dir)

    files = [os.path.join(file_dir, file) for file in os.listdir(file_dir)]

    
    for file in tqdm(files, colour="green"): 
        get_depth_limits(file)
    
    
if __name__ == "__main__": 
    main()