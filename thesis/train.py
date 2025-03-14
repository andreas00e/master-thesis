import os
import argparse

from omegaconf import OmegaConf

import torch
from torch.utils.data import DataLoader
import torch.multiprocessing as mp

import lightning.pytorch as pl 
from lightning.pytorch.trainer import Trainer
from lightning.pytorch.loggers import TensorBoardLogger

from models.autoencoder.autoencoder import DownsampleCVAE

from data.robonet_dataset import RoboNetCustomizedDataset

# Suppress all unwanted tensorflow INFO, WARNING, and ERRORS messages
import tensorflow as tf 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_LAUNCH_BLOCKING']= '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'


def get_train_val_loader(dataset, **dataloader_kwargs):
    train_ds, val_ds = dataset.split_train_val(train_ratio=0.98)
    train_loader = DataLoader(dataset=train_ds, **dataloader_kwargs, shuffle=True)
    val_loader = DataLoader(dataset=val_ds, **dataloader_kwargs, shuffle=False)
    return train_loader, val_loader


def preprocess_config(config, args):
    # overriding horizon
    config.horizon = args.horizon 
    config.model.kwargs.model_kwargs.horizon = args.horizon
    config.dataset.kwargs.horizon = args.horizon

    # avoid gpu rank overflow
    device_count = torch.cuda.device_count()
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

def main(): 
    mp.set_start_method("spawn", force=True)
    torch.cuda.empty_cache()
    
    args = get_parser_args()
    raw_config = OmegaConf.load(f'configs/train.yaml')
    OmegaConf.resolve(raw_config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    path = '~/ehrensberger/RoboNetCustomized/hdf5/' 
    path = os.path.expanduser(path)
    robots = ['kuka'] 
    data = RoboNetCustomizedDataset(path=path, robots=robots)
    train_dataset, test_dataset = torch.utils.data.random_split(data, [0.8, 0.2])

    # Setting `persistent_workers=True` in 'val_dataloader' to speed up the dataloader worker initialization.
    train_loader = DataLoader(dataset=train_dataset, batch_size=8, num_workers=19, persistent_workers=True)
    val_loader = DataLoader(dataset=test_dataset, batch_size=8, num_workers=19, persistent_workers=True)

    config = preprocess_config(raw_config, args)

    epoch_length = len(train_loader) // len(config.trainer.kwargs.devices)
    config.model.kwargs.training_kwargs['num_training_steps'] = epoch_length * config.trainer.kwargs.max_epochs

    pl.seed_everything(raw_config.seed)
    # TODO: Try to stay closer to github implementation 
    
    model: pl.LightningModule = DownsampleCVAE(config.model.kwargs.model_kwargs, config.joint_attention_encoder.kwargs.model_kwargs, 
                                               config.model.kwargs.training_kwargs, mode='pretraining')
    model.to(device)

    logger = TensorBoardLogger(save_dir="logs/")
    trainer = Trainer(default_root_dir="logs/", logger=logger, max_epochs=400)
    # TODO: get arguments from config file 
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)

if __name__ == '__main__':
    main() 