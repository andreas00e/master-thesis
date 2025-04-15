import os
import copy
import argparse

from omegaconf import OmegaConf
from typing import Dict, List#
from lightning.pytorch.callbacks import DeviceStatsMonitor

import torch
from torch.utils.data import DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence 
import torch.multiprocessing as mp

import lightning.pytorch as pl 
from lightning.pytorch.trainer import Trainer
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger

from models.autoencoder.autoencoder import DownsampleCVAE

from data.robonet_dataset import RoboNetCustomizedDataset

# Suppress all unwanted tensorflow INFO, WARNING, and ERRORS messages
import tensorflow as tf 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_LAUNCH_BLOCKING']= '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'

def get_train_val_loader(dataset, dataloader_kwargs):
    train_ds, val_ds, test_ds = random_split(dataset, lengths=[0.8, 0.1, 0.1])

    train_loader = DataLoader(dataset=train_ds, collate_fn=collate_fn, **dataloader_kwargs, shuffle=True)
    val_loader = DataLoader(dataset=val_ds, collate_fn=collate_fn,  **dataloader_kwargs, shuffle=False)
    test_loader = DataLoader(dataset=test_ds, collate_fn=collate_fn, **dataloader_kwargs, shuffle=False)

    return train_loader, val_loader, test_loader


def preprocess_config(config, args):
    # overriding horizon
    config.horizon = args.horizon 
    config.model.kwargs.model_kwargs.horizon = args.horizon
    config.dataset.kwargs.horizon = args.horizon

    # avoid gpu rank overflow
    device_count = torch.cuda.device_count() # 1 
    if len(config.trainer.kwargs.devices) > device_count:
        config.trainer.kwargs.devices = list(range(device_count))
        # print(f'using {device_count} devices')

    return config

def get_parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_name', default='downsample_cvae')
    parser.add_argument('--devices', type=str, default='0')
    parser.add_argument('--horizon', type=int, default=16)

    return parser.parse_args()

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

def main(): 
    mp.set_start_method("spawn", force=True)
    torch.cuda.empty_cache()
    
    args = get_parser_args()
    raw_config = OmegaConf.load(f'configs/train.yaml')
    OmegaConf.resolve(raw_config)
    config = preprocess_config(raw_config, args)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    path = '~/ehrensberger/RoboNetCustomized/hdf5/' 
    path = os.path.expanduser(path)
    robots = ['kuka', 'franka', 'baxter', 'widowx'] 
    # robots = ['kuka', 'widowx']
    data = RoboNetCustomizedDataset(load_dir=path, robots=robots)
    train_loader, val_loader, test_loader = get_train_val_loader(dataset=data, dataloader_kwargs=config.dataloader)
    
    # for i, data in enumerate(train_loader): 
    #     actions = data['actions'].squeeze(-1)
    #     state = data['states']
    #     print(actions.shape)
    #     print(state.shape)
    #     print(actions)
    #     # print(actions)
    #     print('---------------------------------------')
    #     print('\t')
    #     if i == 100: 
    #         break 
    
    # exit() 




    epoch_length = len(train_loader) // len(config.trainer.kwargs.devices)
    config.model.kwargs.training_kwargs['num_training_steps'] = epoch_length * config.trainer.kwargs.max_epochs

    pl.seed_everything(raw_config.seed)
    # TODO: Try to stay closer to github implementation 
    
    model: pl.LightningModule = DownsampleCVAE(config.model.kwargs.model_kwargs, config.joint_attention_encoder.kwargs.model_kwargs, 
                                               config.model.kwargs.training_kwargs, mode='pretraining')
    model.to(device)

    # logger = TensorBoardLogger(save_dir="logs/")
    wandb_logger = WandbLogger(save_dir='logs/')

    trainer = Trainer(default_root_dir="logs/", logger=wandb_logger, callbacks=[DeviceStatsMonitor()], max_epochs=400, profiler='simple')
    # TODO: get arguments from config file 
    # Train model
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    # Test model
    # trainer.test(model=model, dataloaders=test_loader, ckpt_path='logs/lightning_logs/version_5/checkpoints/epoch=99-step=3000.ckpt')
    # trainer.predict(model=model, dataloaders=test_loader, ckpt_path='logs/lightning_logs/version_5/checkpoints/epoch=99-step=3000.ckpt')
    

if __name__ == '__main__':
    main() 