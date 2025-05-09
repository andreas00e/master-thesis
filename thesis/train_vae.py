import os
import argparse

from omegaconf import OmegaConf
from typing import Dict, List
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
from data.mimicgen.data_mimicgen import MimicgenDataset

from lightning.pytorch.callbacks import DeviceStatsMonitor, EarlyStopping, ModelCheckpoint

from pathlib import Path


# Suppress all unwanted tensorflow INFO, WARNING, and ERRORS messages
import tensorflow as tf 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_LAUNCH_BLOCKING']= '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'


def get_train_val_loader(dataset, dataloader_kwargs):
    train_ds, val_ds = random_split(dataset, lengths=dataloader_kwargs.lengths)

    del dataloader_kwargs['lengths']
    train_loader = DataLoader(dataset=train_ds, **dataloader_kwargs, shuffle=True)
    val_loader = DataLoader(dataset=val_ds, **dataloader_kwargs, shuffle=False)

    return train_loader, val_loader

def preprocess_config(config, args):
    # Override horizon of config YAML file
    config.horizon = args.horizon 
    config.model.model_kwargs.horizon = args.horizon
    config.dataset.horizon = args.horizon

    # Avoid GPU rank overflow
    device_count = torch.cuda.device_count() # Only one device available here
    if len(config.trainer.devices) > device_count:
        config.trainer.devices = list(range(device_count))
        print(f'Using {device_count} devices')

    return config

def get_parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=Path, default='thesis/configs/train_vae.yaml', help='Path to config yaml storing parameters for training')
    parser.add_argument('--devices', type=str, default='0', help='Number/Name(s) of GPU(s) available for tria')
    parser.add_argument('--horizon', type=int, default=16, help='Number of actions steps predicited by the VAE')

    return parser.parse_args()

# # Construction of batches due to different number of joints (different lengths of e.g., joint position and joint velocity)
# def collate_fn(data: List[Dict]) -> Dict:
#     batch = dict() 

#     for key, value in data[0].items():
#         sample = [d[key] if isinstance(value, str) else torch.tensor(d[key]) for d in data]
#         if isinstance(value, str): 
#             batch.update({key: sample.copy()})
#         else: 
#             batch.update({key: pad_sequence(sample.copy(), batch_first=True)}) 
#         sample.clear()

#     return batch


def main(): 
    # Ensure compatibility and safety by setting the multiprocessing start method to "spawn"
    mp.set_start_method("spawn", force=True)
    # Release all unocuppied cached memory currently held by caching allocator
    torch.cuda.empty_cache()

    args = get_parser_args()
    raw_config = OmegaConf.load(args.config_path)
    OmegaConf.resolve(raw_config)
    config = preprocess_config(raw_config, args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pl.seed_everything(raw_config.seed)
     
    # Get rid of path here
    path = '~/ehrensberger/mimicgen/datasets/core'
    path = os.path.expanduser(path)
    task = ['stack_d0.hdf5', 'stack_d1.hdf5']
    
    dataset = MimicgenDataset(
        load_dir=path,
        robot='', 
        task=task,
        view='robot0_eye_in_hand_image',
        horizon=16, 
        overlap=False
        )
    
    d = len(dataset)
    
    train_loader, val_loader = get_train_val_loader(dataset=dataset, dataloader_kwargs=config.dataloader)

    epoch_length = len(train_loader) // len(config.trainer.devices)
    config.model.training_kwargs['num_training_steps'] = epoch_length * config.trainer.max_epochs

    model: pl.LightningModule = DownsampleCVAE(
        config.model.model_kwargs, 
        config.model.training_kwargs, 
        mode='pretraining'
        )
    
    model.to(device)

    wandb_logger = WandbLogger(
        name='pretrain_vae', 
        save_dir='/home/ubuntu/ehrensberger/master-thesis/master-thesis/thesis/logs', 
        version='kl_annealing',
        project='pretrain_vae',
        log_model='all'
    )

    # Monitor and log device statistics (e.g., CPU and GPU usage) during training
    device_stats_monitor = DeviceStatsMonitor(
        cpu_stats=True
        )
    
    # Monitor the total validation loss of the VAE and stop training when it stops improving 
    early_stopping = EarlyStopping(
        monitor='val/ae_total_loss', 
        min_delta=1e-4, 
        patience=10, 
        verbose=True, 
        mode='min', 
        check_on_train_epoch_end=True
        )
   
    # Save the model periodically by monitoring the total validation loss of the VAE 
    model_checkpoint = ModelCheckpoint(
        # dirpath='thesis/checkpoints', 
        filename='best_vae', 
        monitor='val/ae_total_loss', 
        verbose=True,
        save_last=True,
        save_top_k=1, 
        mode='min',
        every_n_epochs=5,       
        )

    trainer = Trainer(
        default_root_dir='logs/', 
        logger=wandb_logger, 
        callbacks=[device_stats_monitor, early_stopping, model_checkpoint], 
        max_epochs=400, 
        profiler='simple'
        )

    trainer.fit(
        model=model, 
        train_dataloaders=train_loader, 
        val_dataloaders=val_loader
        )


if __name__ == '__main__':
    main() 