import hydra
from hydra.utils import instantiate

import multiprocessing

import lightning.pytorch as pl


@hydra.main(config_path="cfgs/", config_name="discover_skills", version_base=None)
def main(cfg): 
    multiprocessing.set_start_method("spawn", force=True)
    
    pl.seed_everything(cfg.seed)

    datamodule = instantiate(cfg.datamodule)
    model = instantiate(cfg.model)
    logger = instantiate(cfg.logger)

    trainer = pl.Trainer(logger=logger, **cfg.trainer)
    _ = trainer.fit(model=model, datamodule=datamodule)


if __name__ == "__main__": 
    main()