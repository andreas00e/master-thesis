import os 
import yaml
from omegaconf import OmegaConf
import sys
sys.path.append('/home/ubuntu/ehrensberger/master-thesis/master-thesis')  # Add the parent directory of 'thesis' to sys.path
import lightning.pytorch as pl 
import torch

from torch.utils.data import DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
from thesis.data.robonet_dataset import RoboNetCustomizedDataset

from typing import Dict, List

from lightning.pytorch.loggers import WandbLogger

from RoLD.models.diffusion.downsample_obs_ldm import DownsampleObsLDM

# TODO: Use multiprocessing/threading to speed up function execution 
def collate_fn(data: List[Dict]) -> Dict:
    batch = dict() 

    for key, value in data[0].items():
        sample = [d[key] if isinstance(value, str) else torch.tensor(d[key]) for d in data]
        if isinstance(value, str): 
            batch.update({key: sample.copy()})
        else: 
            batch.update({key: pad_sequence(sample.copy(), batch_first=True)}) 
        sample.clear()

    return batch

def get_dataloader(dataset, dataloader_kwargs): 
    train_set, val_set, test_set = random_split(dataset=dataset, lengths=[0.8, 0.1, 0.1])
    train_loader = DataLoader(dataset=train_set, collate_fn=collate_fn, **dataloader_kwargs)
    val_loader = DataLoader(dataset=val_set, collate_fn=collate_fn, **dataloader_kwargs)
    
    return train_loader, val_loader

def main(): 
    torch.manual_seed(42)
    print('Hello')
    # TODO: Move path to config file 
    ae_ckpt_path = '/home/ubuntu/ehrensberger/master-thesis/master-thesis/thesis/logs/lightning_logs/version_24/checkpoints/epoch=131-step=82236.ckpt'
    path = '/home/ubuntu/ehrensberger/RoLD/RoLD/configs/downsample_obs_ldm.yaml'
    # with open(path) as file: 
    #     try:
    #         config = yaml.safe_load(file)
    #     except yaml.YAMLError as e: 
    #         print(e)

    config = OmegaConf.load(path)
    torch.manual_seed(config.seed)


    config.model.kwargs.model_kwargs.horizon = 16 
    config.model.kwargs.model_kwargs.ckpt_path = None  
    model_kwargs = config.model.kwargs.model_kwargs

    training_kwargs = config.model.kwargs.training_kwargs 
    noise_scheduler_kwargs = config.model.kwargs.noise_scheduler_kwargs
    
    trainer_kwargs = config.trainer.kwargs
    trainer_kwargs['devices'] = [0]
    del trainer_kwargs['logger']
    del trainer_kwargs['pretrain_max_epochs']
    dataset_kwargs = config.dataset
    dataloader_kwargs = config.dataloader

    
    # print(config.keys())
    # exit()

    # config.model.kwargs.model_kwargs.horizon = 16 

    # model_config = config.model.kwargs
    # training_kwargs = config.model.training_kwargs
    # noise_scheduler_config =  config.model.noise_scheduler_kwargs
    # trainer_config = config.trainer.kwargs

    # del training_kwargs['pretrain_max_epochs']
    # training_kwargs['num_training_steps'] = 400
    # trainer_config['devices'] = [0]
    # del trainer_config['logger']
    # trainer_config['logger'] = 'thesis/logs/lightning_logs/diffusion/'

    # config['model']['kwargs']['model_kwargs']['horizon'] = 16 
    # config['model']['kwargs']['model_kwargs']['ckpt_path'] = None

    # config = config['model']['kwargs']
    # model_kwargs = config['model_kwargs']
    # training_kwargs = config['training_kwargs'] 
    # noise_scheduler_kwargs = config['noise_scheduler_kwargs']

    mode = 'pretraining'
    
    load_dir = '~/ehrensberger/RoboNetCustomized/hdf5/'
    load_dir = os.path.expanduser(load_dir)
    robots = ['franka', 'kuka']
    dataset = RoboNetCustomizedDataset(load_dir=load_dir,robots=robots)
    train_loader, val_loader = get_dataloader(dataset, dataloader_kwargs)

    training_kwargs['num_training_steps'] = len(train_loader) * trainer_kwargs['max_epochs']

    


    model: pl.LightningModule = DownsampleObsLDM(ae_ckpt_path=ae_ckpt_path, model_kwargs=model_kwargs,  
                                                 training_kwargs=training_kwargs, noise_scheduler_kwargs=noise_scheduler_kwargs, 
                                                 mode=mode)

    wandb_logger = WandbLogger(log_model="all")

    # epoch_length = len(train_loader)
    # trainer_config['num_training_steps'] = epoch_length * trainer_config.max_epochs
    trainer: pl.Trainer = pl.Trainer(logger=wandb_logger, **trainer_kwargs)

    
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    

    print('Success!')

if __name__ == '__main__': 
    main()