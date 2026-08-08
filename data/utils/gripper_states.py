import os 
import h5py 
import numpy as np 
import pandas as pd
from tqdm import tqdm 


def main(): 
    file_dir = "/home/bing_TUM/ehrensberger/master-thesis/imports/mimicgen/datasets/robot"
    files = [file for file in os.listdir(file_dir) if ("depth" not in file and file.endswith(".hdf5"))]

    robots = ["iiwa", "panda", "sawyer", "ur5e"]
    tasks = ["square", "threading"]

    df = {}
    for robot in robots: 
        df[robot] = {}
        for task in tasks: 
            if robot in ["panda", "sawyer"]: 
                df[robot][task] = {
                    "min": np.array([10.0]*2),
                    "max": np.array([-10.0]*2)
                }
            else:  # iiwa, ur5e
                df[robot][task] = {
                    "min": np.array([10.0]*6),
                    "max": np.array([-10.0]*6)
                }

    for file in tqdm(files): 
        with h5py.File(os.path.join(file_dir, file), "r") as hf: 
            task = file.split("_")[0]
            robot = file.split(".")[0].split("_")[-1]
            
            data = hf["data"]
            for demo in tqdm(data.keys(), leave=False): 
                gripper_qpos = data[demo]["obs"]["robot0_gripper_qpos"][()]  # (n, d)
                
                demo_min = np.min(gripper_qpos, axis=0)  # (d,)
                demo_max = np.max(gripper_qpos, axis=0)  # (d,)

                min_mask = demo_min < df[robot][task]["min"]
                max_mask = demo_max > df[robot][task]["max"]         
                
                if min_mask.any(): 
                    df[robot][task]["min"][min_mask] = demo_min[min_mask]
                if max_mask.any(): 
                    df[robot][task]["max"][max_mask] = demo_max[max_mask]

    rows = []
    for robot in robots:
        for task in tasks:
            row = {"robot": robot, "task": task}
            d = len(df[robot][task]["min"])
            for i in range(d):
                row[f"min_{i}"] = df[robot][task]["min"][i]
                row[f"max_{i}"] = df[robot][task]["max"][i]
            rows.append(row)
    
    pd.DataFrame(rows).to_csv("gripper_state.csv", index=False)
    print("Saved gripper_state.csv")
    
if __name__ == "__main__": 
    main()