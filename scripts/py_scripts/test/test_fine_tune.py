import hydra
import multiprocessing
from r3m import load_r3m
from peft import PeftModel
from hydra.utils import instantiate

import lightning.pytorch as pl

from scripts.py_scripts.setup_environment import setup_environment
from models.fine_tune.fine_tuner import FineTunerVisual
setup_environment()


@hydra.main(config_path="../../../cfgs/", config_name="fine_tune", version_base=None)
def main(cfg):    
    pl.seed_everything(cfg.seed, workers=True)
    
    datamodule = instantiate(cfg.datamodule)
    
    model = FineTunerVisual.load_from_checkpoint(
        "outputs/checkpoints/fine_tune/best-checkpoint-epoch=47-val_loss=0.00.ckpt", 
        strict=False
        )
    model.eval()  
    
    callbacks = [instantiate(callback_cfg) for callback_cfg in cfg.callbacks.values()]
    trainer = instantiate(cfg.trainer, logger=logger, callbacks=callbacks)
    
    trainer.test(model=model, datamodule=datamodule)
    
if __name__ == "__main__": 
    try:
        multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass
        
    main()