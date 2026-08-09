import os 
import pandas as pd 

robots = ["iiwa", "panda", "sawyer", "ur5e"]
robot = "iiwa"

file_dir = "/home/bing_TUM/ehrensberger/master-thesis/thesis/data/metadata/gripper_state_robot.csv"

df = pd.read_csv(file_dir)

cols = [col for col in df.columns if robot in col]

print(df["iiwa_min"].values)
