import hydra
import mimicgen 
import numpy as np
import robosuite as suite
from omegaconf import OmegaConf  # <-- 1. Importieren


@hydra.main(config_name="run_simulation", config_path="../cfgs/", version_base=None)
def main(cfg): 
    env_kwargs = OmegaConf.to_object(cfg.env_kwargs)
    controller_config = suite.load_controller_config(default_controller="OSC_POSE")
    
    env = suite.make(
        controller_configs=controller_config, 
        **env_kwargs
        )

    obs = env.reset()

    for _ in range(1000):
        env.render()  
        
        # Hier eine zufällige kleine Bewegung generieren:
        random_translation = np.random.uniform(-0.05, 0.05, 3)
        random_rotation = np.random.uniform(-0.1, 0.1, 3)
        gripper_action = np.array([1.0]) # Greifer offen halten
        
        action = np.concatenate([random_translation, random_rotation, gripper_action])

        obs, reward, done, info = env.step(action)

        if done:
            obs = env.reset()

    env.close()
    
if __name__ == "__main__": 
    main()