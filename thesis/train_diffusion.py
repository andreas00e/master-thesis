import os 
import sys
sys.path.append('/home/ubuntu/ehrensberger/master-thesis/master-thesis') # Add the parent directory of 'thesis' to sys.path
from omegaconf import OmegaConf

import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, random_split

import lightning.pytorch as pl 
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import DeviceStatsMonitor, EarlyStopping, ModelCheckpoint, RichProgressBar
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBarTheme

from data.mimic.data_mimicgen import MimicgenDataset
from models.diffusion.diffuser import DownsampleObsLDM


def get_dataloader(dataset, dataloader_kwargs): 
    train_set, val_set, _ = random_split(dataset=dataset, lengths=dataloader_kwargs.lengths)
    del dataloader_kwargs['lengths']
    train_loader = DataLoader(dataset=train_set, **dataloader_kwargs, shuffle=True)
    val_loader = DataLoader(dataset=val_set, **dataloader_kwargs, shuffle=False)
    return train_loader, val_loader

def main(): 
    mp.set_start_method("spawn", force=True) # Ensure compatibility and safety by setting the multiprocessing start method to "spawn"
    torch.cuda.empty_cache() # Release all unocuppied cached memory currently held by caching allocator
    
    # TODO: Move path to config file 
    ae_ckpt_path = '/home/ubuntu/ehrensberger/master-thesis/master-thesis/thesis/logs/forward_10_depth_views/free_bits_annealing/checkpoints/best_vae.ckpt'
    path = '/home/ubuntu/ehrensberger/master-thesis/master-thesis/thesis/configs/train_diffusion.yaml'

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
    
    data_dir = '~/ehrensberger/mimicgen/datasets/core'
    data_dir = os.path.expanduser(data_dir)
    
    dataset = MimicgenDataset(
        load_dir=data_dir,
        task=['stack_d0_depth.hdf5', 'stack_d1_depth.hdf5'],
        robot=None, 
        action_horizon=16, # Number of actions to be predicted
        image_horizon=10, # Number of past images given to network as input, only necessary if obseravtions='backwards' or 'both'
        observations='forward', 
        expand_depth='colormap',
    )

    train_loader, val_loader = get_dataloader(dataset, dataloader_kwargs)

    training_kwargs['num_training_steps'] = len(train_loader) * trainer_kwargs['max_epochs']
    
    model: pl.LightningModule = DownsampleObsLDM(ae_ckpt_path=ae_ckpt_path, model_kwargs=model_kwargs,  
                                                 training_kwargs=training_kwargs, noise_scheduler_kwargs=noise_scheduler_kwargs, 
                                                 mode=mode)
    
    # Log diffusion training statistics and save best model checkpoint
    wandb_logger = WandbLogger(
        name='train diffusion',
        save_dir='/home/ubuntu/ehrensberger/master-thesis/master-thesis/thesis/logs/',
        version='free_bits_annealing', 
        project='forward_10_depth_views_diffusion', # 'forward_10_depth_views', # 'single_depth_views'
        log_model="all"
        )

    # Monitor and log device statistics (e.g., CPU and GPU usage) during training
    device_stats_monitor = DeviceStatsMonitor(
        cpu_stats=True
    )

    # Monitor the total denoise loss of the diffusion model and stop training when it stops improving 
    early_stopping = EarlyStopping(
        monitor='val/denoise_loss', 
        min_delta=1e-4, 
        patience=10, 
        verbose=True, 
        mode='min',
        check_on_train_epoch_end=False 
        )
    
    # Save the model periodically by monitoring the total validation loss of the VAE 
    model_checkpoint = ModelCheckpoint(
        filename='best_diffusion',
        monitor='train/denoise_loss', 
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
        profiler='simple',
        **trainer_kwargs
        )

    trainer.fit(
        model=model, 
        train_dataloaders=train_loader, 
        val_dataloaders=val_loader
        )


if __name__ == '__main__': 
    main()