import os
import hydra
import warnings
import multiprocessing
from termcolor import colored, cprint
from omegaconf import OmegaConf
from hydra.utils import instantiate

import lightning.pytorch as pl

from scripts.py_scripts.setup_environment import setup_environment
setup_environment()


@hydra.main(config_path="../../cfgs/", config_name="fine_tune", version_base=None)
def main(cfg):     
    pl.seed_everything(cfg.seed, workers=True)
    
    datamodule = instantiate(cfg.datamodule)
    model = instantiate(cfg.model_rgb)
    logger = instantiate(cfg.logger)
    trainer = instantiate(cfg.trainer, logger=logger)
    
    trainer.fit(model=model, datamodule=datamodule)
    
    model = instantiate(cfg.model_gripper)
    
    trainer.fit(model=model, datamodule=datamodule)
    
    
if __name__ == "__main__": 
    try:
        multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass
        
    main()