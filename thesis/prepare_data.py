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

                
def main(): 
    file_dir = "~/ehrensberger/master-thesis/mimicgen/datasets/robot"
    file_dir = os.path.expanduser(file_dir)
    prepare_hdf5(file_dir)
    
if __name__ == "__main__": 
    main()