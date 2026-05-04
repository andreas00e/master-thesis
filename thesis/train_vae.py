import os
import hydra 
from termcolor import colored

import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, random_split

import lightning.pytorch as pl 
from lightning.pytorch.trainer import Trainer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBarTheme
from lightning.pytorch.callbacks import DeviceStatsMonitor, EarlyStopping, ModelCheckpoint, RichProgressBar

from data.mimicgen.data_mimicgen import MimicgenDataset
from models.autoencoder.autoencoder import DownsampleCVAE

# Suppress all unwanted tensorflow INFO, WARNING, and ERRORS messages
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_LAUNCH_BLOCKING"]= "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"


def get_dataloader(dataset, dataloader):
    train_ds, val_ds, _ = random_split(dataset, lengths=dataloader.lengths)
    
    del dataloader["lengths"]
    train_loader = DataLoader(dataset=train_ds, **dataloader, shuffle=True)
    val_loader = DataLoader(dataset=val_ds, **dataloader, shuffle=False)
    return train_loader, val_loader

def preprocess_config(cfg, args):
    device_count = torch.cuda.device_count() # current server: 1
    if len(cfg.trainer.devices) > device_count:
        cfg.trainer.devices = list(range(device_count))
        if device_count == 1: 
            print(colored(f"Using 1 device", "green"))
        else: 
            print(colored(f"Using {device_count} devices", "green"))      
    return cfg

@hydra.main(config_path="cfgs", config_name="run.yaml", version_base=None)
def main(cfg): 
    mp.set_start_method("spawn", force=True) # ensure compatibility and safety by setting the mp start method to "spawn"
    torch.cuda.empty_cache() # release all unocuppied cached memory currently held by the caching allocator
    
    pl.seed_everything(cfg.seed)
    
    dataset = MimicgenDataset(**cfg.data.dataset)
    train_loader, val_loader = get_dataloader(dataset=dataset, dataloader=cfg.data.datalaoder)

    epoch_length = len(train_loader) // len(cfg.trainer.trainer_devices.trainer.devices)
    cfg.model.training_kwargs.num_training_steps = epoch_length * cfg.trainer.max_epochs 

    model = DownsampleCVAE(model_kwargs=cfg.model.model_kwargs, training_kwargs=cfg.model.training_kwargs, mode=cfg.model.mode) 
    
    wandb_logger = WandbLogger(**cfg.logger) # log VAE training statitistics, save best model 
    
    model_checkpoint = ModelCheckpoint(**cfg.callbacks.model_checkpoint) # save model periodically by monitoring VAEs total val loss
    early_stopping = EarlyStopping(**cfg.callbacks.early_stopping) # monitor VAEs total val loss, stop training when improvements stall 
    device_stats_monitor = DeviceStatsMonitor(**cfg.callbacks.device_stats_monitor) # log device statistics 
    rich_progress_bar = RichProgressBar(theme=RichProgressBarTheme(**cfg.callbacks.rich_progress_bar.theme))
    
    trainer = Trainer(logger=wandb_logger, callbacks=[device_stats_monitor, early_stopping, model_checkpoint, rich_progress_bar], **cfg.trainer)
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)


if __name__ == "__main__":
    main() 