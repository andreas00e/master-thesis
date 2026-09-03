import os 
import warnings
import multiprocessing

import hydra
from hydra.utils import instantiate

import lightning.pytorch as pl

os.environ["OMPI_MCA_btl"] = "^openib" # suppress Open MPI warnings
os.environ["PYOPENGL_PLATFORM"] = "egl"  # for headless OpenGL rendering
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")
warnings.filterwarnings("ignore", category=UserWarning, module="lightning")
warnings.filterwarnings("ignore", category=UserWarning, module="lightning.pytorch.utilities._pytree")


@hydra.main(config_path="cfgs/", config_name="discover_skills", version_base=None)
def main(cfg):     
    multiprocessing.set_start_method("spawn", force=True)
    pl.seed_everything(cfg.seed)

    datamodule = instantiate(cfg.datamodule)
    model = instantiate(cfg.model)
    logger = instantiate(cfg.logger)
    trainer = instantiate(cfg.trainer, logger=logger)
    
    trainer.fit(model=model, datamodule=datamodule)
    

if __name__ == "__main__":     
    main()