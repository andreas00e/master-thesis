import hydra
import multiprocessing
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
        "/home/ehre/Documents/master-thesis/outputs/checkpoints/skill_discovery_new/wo_time_wo_gripper_w_aug_wo_do_nhead_2_four_b_eight_pt_mag_1/best-checkpoint.ckpt", 
        weights_only=False
        )
    
    model.eval() 
    trainer = instantiate(cfg.trainer)
    
    trainer.test(model=model, datamodule=datamodule)
    
if __name__ == "__main__": 
    try:
        multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass
        
    main()