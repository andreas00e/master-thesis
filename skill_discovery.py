import hydra
from hydra.utils import instantiate

import lightning as pl


@hydra.main(config_path="cfgs/", config_name="skill_discovery", version_base=None)
def main(cfg): 
    pl.seed_everything(cfg.seed)

    datamodule = instantiate(cfg.data.datamodule)
    model = instantiate(cfg.model)
    logger = instantiate(cfg.logger)

    trainer = pl.Trainer(logger=logger, **cfg.trainer)
    trainer.fit(model=model, datamodule=datamodule)


if __name__ == "__main__": 
    main()