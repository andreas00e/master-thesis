import cv2
import h5py
import hydra
import random 
from tqdm import tqdm 

import robosuite as suite
from robosuite import load_composite_controller_config

from data.utils.load_files import get_files

@hydra.main(config_path="../cfgs/", config_name="run", version_base=None)
def main(cfg): 
    # controller_config = load_composite_controller_config(controller="BASIC")
    
    # env = suite.make(controller_configs=controller_config, **cfg.env_kwargs)
    # obs = env.reset() 
    
    file = get_files(cfg.data_dir, cfg.robots, cfg.tasks)[0]
    
    with h5py.File(file, "r") as hf: 
        data = hf["data"]
        
        demo = random.choice(list(data.keys()))

        rgb_obs = data[demo]["obs"][cfg.camera][()] # [n, h, w, c]
        
        h, w, _ = rgb_obs[0].shape
        
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_path = f"{cfg.env_kwargs.env_name}_{cfg.camera}_.mp4"
        writer = cv2.VideoWriter(video_path, fourcc, 20.0, (w, h))
        
        for i in tqdm(range(rgb_obs.shape[0])): 
            image = rgb_obs[i]
            
            bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            writer.write(bgr_image)

    # env.close()
    writer.release()
    
    print(f"Saved {video_path}")

if __name__ == "__main__": 
    main()
