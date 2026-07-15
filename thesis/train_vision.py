import hydra 

import lightning as pl
from models.vision.vision import VisionEmbedder 
from data.datamodule.datamodule import MimicGenRobotDataModule


@hydra.main(config_path="cfgs/", config_name="visualization.yaml", version_base=None)
def main(cfg): 
    pl.seed_everything(cfg.seed)

    datamodule = MimicGenRobotDataModule(**cfg.data.datamodule)
    model = VisionEmbedder(**cfg.model)

    trainer = pl.Trainer(**cfg.trainer)
    _ = trainer.test(model=model, datamodule=datamodule)
    

if __name__ == "__main__": 
    main() 