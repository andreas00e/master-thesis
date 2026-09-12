import os
import hydra
import warnings
import multiprocessing
from termcolor import colored, cprint
from omegaconf import OmegaConf
from hydra.utils import instantiate

import lightning.pytorch as pl


OmegaConf.register_new_resolver("eval", eval)

#Suppress Open MPI warnings
os.environ["OMPI_MCA_btl"] = "^openib" 
# Headless Offscreen Rendering 
os.environ["MUJOCO_GL"] = "egl" # Mujoco
os.environ["PYOPENGL_PLATFORM"] = "egl" # OpenGL 
# Distributed Training
os.environ["NCCL_P2P_DISABLE"]="1"
os.environ["NCCL_IB_DISABLE"]="1"

warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")
warnings.filterwarnings("ignore", category=UserWarning, module="r3m")
warnings.filterwarnings("ignore", category=UserWarning, module="lightning")
warnings.filterwarnings("ignore", category=UserWarning, module="lightning.pytorch.utilities._pytree")
warnings.filterwarnings("ignore", message=".*incompatible copy of pydevd already imported.*")


@hydra.main(config_path="../../cfgs/", config_name="fine_tune", version_base=None)
def main(cfg):     
    pl.seed_everything(cfg.seed, workers=True)

    cprint("## :=) ## xD ## ;=) ## :0 ## :=) ## xD ## ;=) ## :0 ## :=) ## xD ## ;=) ## :0 ##", "magenta", "on_black", attrs=["italic", "blink"])
    cprint("Setup training!", "light_magenta", "on_black", attrs=["bold", "blink"])
    cprint("## :=) ## xD ## ;=) ## :0 ## :=) ## xD ## ;=) ## :0 ## :=) ## xD ## ;=) ## :0 ##", "magenta", "on_black", attrs=["italic", "blink"])
    
    datamodule = instantiate(cfg.datamodule)
    model = instantiate(cfg.model)
    logger = instantiate(cfg.logger)
    trainer = instantiate(cfg.trainer, logger=logger)
    
    cprint("## :=) ## xD ## ;=) ## :0 ## :=) ## xD ## ;=) ## :0 ## :=) ## xD ## ;=) ## :0 ##", "magenta", "on_black", attrs=["italic", "blink"])
    cprint("Start training!", "light_magenta", "on_black", attrs=["bold", "blink"])
    cprint("## :=) ## xD ## ;=) ## :0 ## :=) ## xD ## ;=) ## :0 ## :=) ## xD ## ;=) ## :0 ##", "magenta", "on_black", attrs=["italic", "blink"])
    
    trainer.fit(model=model, datamodule=datamodule)
    
    cprint("## :=) ## xD ## ;=) ## :0 ## :=) ## xD ## ;=) ## :0 ## :=) ## xD ## ;=) ## :0 ##", "magenta", "on_black", attrs=["italic", "blink"])
    cprint("Finished training!", "light_magenta", "on_black", attrs=["bold", "blink"])
    cprint("## :=) ## xD ## ;=) ## :0 ## :=) ## xD ## ;=) ## :0 ## :=) ## xD ## ;=) ## :0 ##", "magenta", "on_black", attrs=["italic", "blink"])
    
if __name__ == "__main__": 
    try:
        multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass
        
    main()