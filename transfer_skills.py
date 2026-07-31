import hydra
from hydra.utils import instantiate

import multiprocessing 

import lightning.pytorch as pl 

from data.transfer.datamodule import TransferDataModule


@hydra.main(config_path="cfgs/", config_name="transfer_skills", version_base=None)
def main(cfg):   
    multiprocessing.set_start_method("spawn", force=True)
    
    pl.seed_everything(cfg.seed) 
     
     
    
    
    datamodule = TransferDataModule(**cfg.data.datamodule)
    
    
    model = instantiate(cfg.model)
    logger = instantiate(cfg.logger)

    trainer = pl.Trainer(logger=logger, **cfg.trainer)
    
    if cfg.stage == "train": 
        _ = trainer.fit(model=model, datamodule=datamodule)
    elif cfg.stage == "test": 
        _ = trainer.test(model=model, datamodule=datamodule)
    elif cfg.stage == "predict": 
        _  = trainer.predict(model=model, datamodule=datamodule)
        
if __name__ == "__main__": 
    main() 