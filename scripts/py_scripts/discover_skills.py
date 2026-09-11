import os 
import warnings
import multiprocessing

import hydra
from hydra.utils import instantiate

import lightning.pytorch as pl

os.environ["OMPI_MCA_btl"] = "^openib" # suppress Open MPI warnings

# Headless Offscreen Rendering 
os.environ["MUJOCO_GL"] = "egl" # Mujoco
os.environ["PYOPENGL_PLATFORM"] = "egl" # OpenGL 

# Distributed Training
os.environ["NCCL_P2P_DISABLE"]="1"
os.environ["NCCL_IB_DISABLE"]="1"

warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")
warnings.filterwarnings("ignore", category=UserWarning, module="lightning")
warnings.filterwarnings("ignore", category=UserWarning, module="lightning.pytorch.utilities._pytree")
warnings.filterwarnings("ignore", message=".*incompatible copy of pydevd already imported.*")

@hydra.main(config_path="cfgs/", config_name="discover_skills", version_base=None)
def main(cfg):     
    pl.seed_everything(cfg.seed, workers=True)

    datamodule = instantiate(cfg.datamodule)
    model = instantiate(cfg.model)
    logger = instantiate(cfg.logger)
    trainer = instantiate(cfg.trainer, logger=logger)
    
    trainer.fit(model=model, datamodule=datamodule)
    

if __name__ == "__main__": 
    try:
        multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass
        
    main()