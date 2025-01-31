import os 
from pathlib import Path 
from torch.utils.data import Dataset
import pickle
import numpy as np 
import h5py 
from matplotlib import pyplot as plt 


class Dataset(Dataset):
    def __init__(self, embodiments, path="data/datasets_raw", horizon=16): 

        self.data = []
        self.path = os.path.join(os.getcwd(), path)
        self.embodiments = embodiments
        
        path_embodiments = [x for x in os.listdir(self.path) if x in embodiments]
        
        for embodiment in path_embodiments: 
            for (root, dirs, files) in os.walk(os.path.join(self.path, embodiment)):

                if any(file.endswith(".pkl") for file in files): 
                    obs = os.listdir(os.path.join(root, dirs[0]))
                    with open(os.path.join(root, files[1]), "rb") as f: 
                        try: 
                            data = pickle.load(f)
                            full_state = data["full_state"] # -> np.darray(50, 7)
                            qpos = data["qpos"]
                            # qvel = data["qvel"]

                            for i,o in enumerate(obs):
                                if horizon+1 == len(obs):
                                    break
                                else: 
                                    # data_point = (full_state[i:horizon+1], qpos[i:horizon+1], qvel[i:horizon+1], o, embodiment)
                                    data_point = (full_state[i:horizon+1], qpos[i:horizon+1], o, embodiment)
                                    self.data.append(data_point)
                        except FileNotFoundError: 
                            print("File does not exist!")                                   

                elif any(file.endswith(".h5") for file in files): 
                    for file in files: 
                        try: 
                            with h5py.File(os.path.join(root, file), "r") as f: 
                                ee_cartesian_pos_ob = np.array(f["ee_cartesian_pos_ob"])
                                front_cam_ob = np.array(f["front_cam_ob"])
                                joint_pos_ob = np.array(f["joint_pos_ob"])  
                                
                                for i in range(ee_cartesian_pos_ob.shape[0]): 
                                    if horizon+1 == len(obs):
                                        break
                                    else: 
                                        data_point = (ee_cartesian_pos_ob[i:horizon+1], joint_pos_ob[i:horizon+1], front_cam_ob[i], embodiment)
                                        self.data.append(data_point)
                        except FileNotFoundError: 
                            print("File does not exist!")
    
    def __getitem__(self, index): 
        return self.data[index]

    def __len__(self): 
        return len(self.data)