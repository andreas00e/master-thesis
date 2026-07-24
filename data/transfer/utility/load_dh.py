import hydra 
from omegaconf import DictConfig, OmegaConf

import torch

config_path = "../dh_conf"
config_name = "dh_params"

@hydra.main(config_path=config_path, config_name=config_name, version_base=None)
def main(cfg: DictConfig) -> None: 
    
    cfg_dict = {
        key: torch.stack([torch.tensor(v, dtype=torch.float32) for v in inner.values()]).T
        for key, inner in cfg.items()
    }
    
    print(cfg_dict)
        
if __name__ == "__main__": 
    main()