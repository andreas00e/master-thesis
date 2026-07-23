import hydra 
from datetime import datetime 
from zoneinfo import ZoneInfo

import lightning as pl
from lightning.pytorch.loggers import WandbLogger

from models.discover.discover import SkillDiscovery
from data.discover.datamodule import MimicGenRobotDataModule


@hydra.main(config_path="cfgs/", config_name="skill_discovery.yaml", version_base=None)
def main(cfg): 
    pl.seed_everything(cfg.seed)

    datamodule = MimicGenRobotDataModule(**cfg.data.datamodule)
    model = SkillDiscovery(**cfg.model)
    
    cfg.logger.verion = datetime.now(ZoneInfo("Europe/Berlin")).strftime("%Y_%m_%d_%H_%M")
    logger = WandbLogger(**cfg.logger)

    trainer = pl.Trainer(
        logger=logger, 
        **cfg.trainer
        )
    _ = trainer.test(model=model, datamodule=datamodule)
    

if __name__ == "__main__": 
    main()