import hydra
import multiprocessing
from r3m import load_r3m
from peft import PeftModel
from hydra.utils import instantiate

import lightning.pytorch as pl

from scripts.py_scripts.setup_environment import setup_environment
from models.discover.skill_encoder import SkillEncoder
setup_environment()


@hydra.main(config_path="../../../cfgs/", config_name="discover_skills", version_base=None)
def main(cfg):    
    pl.seed_everything(cfg.seed, workers=True)
    
    datamodule = instantiate(cfg.datamodule)
    
    model = SkillEncoder.load_from_checkpoint(
        "/dss/dssfs04/lwp-dss-0002/pn36ce/pn36ce-dss-0000/ehrensberger/master-thesis/outputs/checkpoints/skill_discovery/best-checkpoint-epoch=10-val_loss=0.00.ckpt"
        )
    
    model.eval() 
    trainer = instantiate(cfg.trainer)
    
    out = trainer.test(model=model, datamodule=datamodule)
    print(f"output type: {type(out)}")
    print("Success")
    
if __name__ == "__main__": 
    try:
        multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass
        
    main()