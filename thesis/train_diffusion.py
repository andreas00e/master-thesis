import os 
import yaml
from omegaconf import OmegaConf
import sys
sys.path.append('/home/ubuntu/ehrensberger/master-thesis/master-thesis')  # Add the parent directory of 'thesis' to sys.path
import lightning.pytorch as pl 
import torch
import torch.multiprocessing as mp

from torch.utils.data import DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence

from thesis.data.mimicgen.data_mimicgen import MimicgenDataset

from typing import Dict, List

from lightning.pytorch.loggers import WandbLogger

from models.diffusion.diffuser import DownsampleObsLDM


from lightning.pytorch.callbacks import DeviceStatsMonitor, EarlyStopping, ModelCheckpoint, RichProgressBar
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
    train_set, val_set, _ = random_split(dataset=dataset, lengths=[0.85, 0.15, 0.0])
    # train_loader = DataLoader(dataset=train_set, collate_fn=collate_fn, **dataloader_kwargs)
    # val_loader = DataLoader(dataset=val_set, collate_fn=collate_fn, **dataloader_kwargs)
    train_loader = DataLoader(dataset=train_set, **dataloader_kwargs)
    val_loader = DataLoader(dataset=val_set, **dataloader_kwargs)
    return train_loader, val_loader

def main(): 
    # Ensure compatibility and safety by setting the multiprocessing start method to "spawn"
    mp.set_start_method("spawn", force=True)
    # Release all unocuppied cached memory currently held by caching allocator
    torch.cuda.empty_cache()
    # TODO: Move path to config file 
    # ae_ckpt_path = '/home/ubuntu/ehrensberger/master-thesis/master-thesis/thesis/logs/lightning_logs/version_24/checkpoints/epoch=131-step=82236.ckpt'
    ae_ckpt_path = '/home/ubuntu/ehrensberger/master-thesis/master-thesis/thesis/logs/pretrain_vae/1/checkpoints/best_vae-v1.ckpt'
    # ae_ckpt_path = '/home/ubuntu/ehrensberger/master-thesis/master-thesis/thesis/logs/pretrain_vae/17ucv23k/checkpoints/last.ckpt'
    # path = '/home/ubuntu/ehrensberger/RoLD/RoLD/configs/downsample_obs_ldm.yaml'
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
    
    # load_dir = '~/ehrensberger/RoboNetCustomized/hdf5/'
    # load_dir = os.path.expanduser(load_dir)
    # robots = ['franka', 'kuka']
     
    path = '~/ehrensberger/mimicgen/datasets/core'
    path = os.path.expanduser(path)
    task = ['stack_d0.hdf5', 'stack_d1.hdf5']
    
    dataset = MimicgenDataset(
        load_dir=path,
        robot='', 
        task=task,
        view='robot0_eye_in_hand_image',
        horizon=16, 
        )
    train_loader, val_loader = get_dataloader(dataset, dataloader_kwargs)

    training_kwargs['num_training_steps'] = len(train_loader) * trainer_kwargs['max_epochs']
    
    model: pl.LightningModule = DownsampleObsLDM(ae_ckpt_path=ae_ckpt_path, model_kwargs=model_kwargs,  
                                                 training_kwargs=training_kwargs, noise_scheduler_kwargs=noise_scheduler_kwargs, 
                                                 mode=mode)
    
    # Log diffusion training statistics and save best model checkpoint
    wandb_logger = WandbLogger(
        name='pretrain_diffusion_views',
        save_dir='/home/ubuntu/ehrensberger/master-thesis/master-thesis/thesis/logs',
        version='best_epoch', 
        project='pretrain_diffusion',
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
        check_on_train_epoch_end=False # check runs at the end of the validation
        )
    
    # Save the model periodically by monitoring the total validation loss of the VAE 
    model_checkpoint = ModelCheckpoint(
        # filename='{epoch:02d}_{val_loss:.2f}')  
        filename='best_diffusion',
        monitor='train/denoise_loss', 
        verbose=True, 
        # save_last=True, 
        save_top_k=1, 
        mode='min', 
        every_n_epochs=5, 
    )        

    trainer = pl.Trainer(
        default_root_dir='/logs',
        logger=wandb_logger, 
        callbacks=[device_stats_monitor, early_stopping, model_checkpoint, RichProgressBar()], 
        profiler='simple',
        **trainer_kwargs
        )

    trainer.fit(
        model=model, 
        train_dataloaders=train_loader, 
        val_dataloaders=val_loader)

    print('Success!')


if __name__ == '__main__': 
    main()