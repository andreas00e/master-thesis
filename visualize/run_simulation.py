import torch 
import hydra
import mimicgen
import numpy as np
import pandas as pd 
from pathlib import Path
import robosuite as suite
from omegaconf import OmegaConf 

from data.utils.transforms import get_transforms
from data.transfer.datamodule import get_joint_dsc

# from models.transfer.sad import SkillConditionedActionDecoder

@hydra.main(config_name="run_simulation", config_path="../cfgs/", version_base=None)
def main(cfg): 
    robot = cfg.robot.lower()
    transforms = get_transforms(cfg.transforms_list)
    joint_dsc = get_joint_dsc(cfg.cfgs_dir)[robot]
    dataframe_gripper = pd.read_csv(Path(cfg.meta_dir) / "gripper_state_robot.csv")
    
    env_kwargs = OmegaConf.to_object(cfg.env_kwargs)
    controller_config = suite.load_controller_config(default_controller="OSC_POSE")
    
    env = suite.make(
        controller_configs=controller_config, 
        **env_kwargs
        )

    obs = env.reset()
    
    for _ in range(1000):
        env.render()  
        
        rgb_one = obs["robot0_eye_in_hand_image"][()] # [height, width, channels]
        rgb_one = torch.from_numpy(rgb_one).permute(2, 0, 1) # [channels, height, width]
        rgb_one = transforms(rgb_one)[None, ...] # [1, channels, height, width]
        
        rgb_two = obs["agentview_image"][()] # [height, width, channels]
        rgb_two = torch.from_numpy(rgb_two).permute(2, 0, 1) # [channels, height, width]
        rgb_two = transforms(rgb_two)[None, ...] # [1, channels, height, width]
        
        g_qpos = obs["robot0_gripper_qpos"][()]
        min_col, max_col = f"{robot}_min", f"{robot}_max"
        g_min = dataframe_gripper[min_col].values[:g_qpos.shape[-1]] # [d]
        g_max = dataframe_gripper[max_col].values[:g_qpos.shape[-1]] # [d]
        g_qpos = np.clip((g_qpos - g_min) / ((g_max - g_min) + 1e-8), 0.0, 1.0) # [num_samples, condition_horizon, d] 
        g_qpos = np.mean(g_qpos, axis=-1) # [num_sampoles, condition_horizon]
        g_qpos = torch.tensor(g_qpos, dtype=torch.float32) # []
        
        joint_pos_cos = obs["robot0_joint_pos_cos"]
        joint_pos_sin = obs["robot0_joint_pos_sin"]
        joint_pos = np.arctan2(joint_pos_cos, joint_pos_sin)
        joint_pos = torch.from_numpy(joint_pos) 

        joint_vel = obs["robot0_joint_vel"][()]
        joint_vel = torch.from_numpy(joint_vel)
        
        joint_obs = torch.stack(tensors=(joint_pos, joint_vel), dim=-1).to(torch.float32) # [joints, 2]   
        print(f"joint_obs: {joint_obs}")

        random_translation = np.random.uniform(-0.05, 0.05, 3)
        random_rotation = np.random.uniform(-0.1, 0.1, 3)
        gripper_action = np.array([1.0]) # keep gripper open
        action = np.concatenate([random_translation, random_rotation, gripper_action])

        obs = env.step(action)[0]

    env.close()
    
if __name__ == "__main__": 
    main()