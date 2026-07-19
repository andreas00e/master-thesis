import hydra 
import lightning as pl

from models.discover.discover import SkillDiscovery
from data.discover.datamodule import MimicGenRobotDataModule


@hydra.main(config_path="cfgs/", config_name="skill_discovery.yaml", version_base=None)
def main(cfg): 
    pl.seed_everything(cfg.seed)

    datamodule = MimicGenRobotDataModule(**cfg.data.datamodule)
    model = SkillDiscovery(**cfg.model)

    trainer = pl.Trainer(**cfg.trainer)
    _ = trainer.test(model=model, datamodule=datamodule)
    

if __name__ == "__main__": 
    main()