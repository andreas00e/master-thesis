import cv2
import h5py
import hydra
import random 
import robosuite as suite
from robosuite.controllers import load_controller_config

from data.utils.load_files import get_files

@hydra.main(config_path="cfgs/", config_name="run", version_base=None)
def main(cfg): 
    # Create environment 
    controller_config = load_controller_config(default_controller="OSC_POSE")
    env = suite.make(controller_configs=controller_config, **cfg.env_kwargs)
    obs = env.reset() # reset the environment

    # Define codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(f"{cfg.env_kwargs.env_name}_{cfg.camera}_.mp4", fourcc, 20.0, (1080, 1080))
    
    file = get_files(cfg.data_dir, cfg.robots, cfg.tasks)[0]
    
    with h5py.File(file, "r") as hf: 
        data = hf["data"]
        demo = random.shuffle(list(data.keys()))[0]
        
        rgb_obs = data[demo]["obs"][cfg.camera][()] # [n, h, w, c]
        for i in range(rgb_obs.shape[0]): 
            image = rgb_obs[i]
            writer.write(image)

    env.close()
    writer.release()
    
    print(f"Saved {cfg.env_kwargs.env_name}_{cfg.camera}_.mp4")

if __name__ == "__main__": 
    main() 