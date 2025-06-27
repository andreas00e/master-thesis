import os
import argparse
from pathlib import Path
from omegaconf import OmegaConf
from termcolor import colored

import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, random_split

import lightning.pytorch as pl 
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import DeviceStatsMonitor, EarlyStopping, ModelCheckpoint, RichProgressBar
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBarTheme

from data.mimic.data_mimicgen import MimicgenDataset
from models.autoencoder.autoencoder import DownsampleCVAE

# Suppress all unwanted tensorflow INFO, WARNING, and ERRORS messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_LAUNCH_BLOCKING']= '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'


def get_train_val_loader(dataset, dataloader_kwargs):
    train_set, val_set, _ = random_split(dataset, lengths=dataloader_kwargs.lengths)
    del dataloader_kwargs['lengths']
    train_loader = DataLoader(dataset=train_set, **dataloader_kwargs, shuffle=True)
    val_loader = DataLoader(dataset=val_set, **dataloader_kwargs, shuffle=False)

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
        if device_count == 1: 
            print(colored(f'Using 1 device', 'green'))
        else: 
            print(colored(f'Using {device_count} devices', 'green'))      

    return config

def get_parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=Path, default='thesis/configs/train_vae.yaml', help='Path to config yaml storing parameters for training')
    parser.add_argument('--data_dir', type=Path, default='~/ehrensberger/mimicgen/datasets/core', help='Path to dataset containing actions, images, etc.')
    parser.add_argument('--devices', type=str, default='0', help='Number/Name(s) of GPU(s) available for tria')
    parser.add_argument('--horizon', type=int, default=16, help='Number of actions steps predicited by the VAE')

    return parser.parse_args()


def main(): 
    mp.set_start_method("spawn", force=True) # Ensure compatibility and safety by setting the multiprocessing start method to "spawn"
    torch.cuda.empty_cache() # Release all unocuppied cached memory currently held by caching allocator

    args = get_parser_args()
    raw_config = OmegaConf.load(args.config_path)
    OmegaConf.resolve(raw_config)
    config = preprocess_config(raw_config, args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pl.seed_everything(raw_config.seed)
    data_dir = os.path.expanduser(args.data_dir)
    
    dataset = MimicgenDataset(
        load_dir=data_dir,
        task=['stack_d0_depth.hdf5', 'stack_d1_depth.hdf5'],
        robot=None, 
        action_horizon=16, # Number of actions to be predicted
        image_horizon=10, # Number of past images given to network as input, only necessary if obseravtions='backwards' or 'both'
        observations='forward', 
        expand_depth='colormap'
        )
    
    train_loader, val_loader = get_train_val_loader(dataset=dataset, dataloader_kwargs=config.dataloader)

    epoch_length = len(train_loader) // len(config.trainer.devices)
    config.model.training_kwargs['num_training_steps'] = epoch_length * config.trainer.max_epochs

    model: pl.LightningModule = DownsampleCVAE(
        config.model.model_kwargs, 
        config.model.training_kwargs, 
        mode='pretraining'
        )
    
    model.to(device)
    
    # Log variational autoencoder training statistics and save best model checkpoint
    wandb_logger = WandbLogger(
        name='train vae', 
        save_dir='/home/ubuntu/ehrensberger/master-thesis/master-thesis/thesis/logs/', 
        version='free_bits_annealing',
        project= 'test', # 'forward_10_depth_views', # 'single_depth_views'
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
        filename='best_vae', 
        monitor='val/ae_total_loss', 
        verbose=True,
        save_top_k=1, 
        mode='min',
        every_n_epochs=5,       
        )
    
    richProgressBar = RichProgressBar(
        theme=RichProgressBarTheme(
            description="green_yellow",
            progress_bar="green1",
            progress_bar_finished="green1",
            progress_bar_pulse="#6206E0",
            batch_progress="green_yellow",
            time="grey82",
            processing_speed="grey82",
            metrics="grey82",
            metrics_text_delimiter="\n",
            metrics_format=".3e",
            )
        )  

    trainer = pl.Trainer(
        default_root_dir='logs/', 
        logger=wandb_logger, 
        callbacks=[device_stats_monitor, early_stopping, model_checkpoint, richProgressBar], 
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