import os 
import warnings
import multiprocessing

import hydra
from hydra.utils import instantiate

import lightning.pytorch as pl

os.environ["OMPI_MCA_btl"] = "^openib" # suppress Open MPI warnings
os.environ["PYOPENGL_PLATFORM"] = "egl"  # for headless OpenGL rendering
warnings.filterwarnings("ignore", category=UserWarning, module="lightning")


@hydra.main(config_path="cfgs/", config_name="transfer_skills", version_base=None)
def main(cfg): 
    multiprocessing.set_start_method("spawn", force=True)
    
    pl.seed_everything(cfg.seed)

    datamodule = instantiate(cfg.datamodule)
    model = instantiate(cfg.model)
    logger = instantiate(cfg.logger)

    trainer = pl.Trainer(logger=logger, **cfg.trainer)
    _ = trainer.fit(model=model, datamodule=datamodule)
    
    
    print("Hello")

if __name__ == "__main__": 
    main()