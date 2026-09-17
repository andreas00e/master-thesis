import hydra
import multiprocessing
import lightning.pytorch as pl
from hydra.utils import instantiate

from scripts.py_scripts.setup_environment import setup_environment
setup_environment()

@hydra.main(config_path="../../cfgs", config_name="discover_skills", version_base=None)
def main(cfg):     
    pl.seed_everything(cfg.seed, workers=True)

    datamodule = instantiate(cfg.datamodule)
    model = instantiate(cfg.model)
    logger = instantiate(cfg.logger)
    
    callbacks = [instantiate(callback_cfg) for callback_cfg in cfg.callbacks.values()]
    trainer = instantiate(cfg.trainer, logger=logger, callbacks=callbacks)
    
    trainer.fit(model=model, datamodule=datamodule)
    

if __name__ == "__main__":     
    try:
        multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass
        
    main()