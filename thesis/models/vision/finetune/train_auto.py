import torch
from torch.utils.data import DataLoader, random_split

import lightning.pytorch as pl 
from lightning.pytorch.trainer import Trainer 
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint


from models.vision.finetune.auto import AutoEncoder, AutoEncoderR3MOnly
from models.vision.finetune.data_auto import FinetuneR3MData


def get_dataloaders(dataset):
    # train_set, val_set, _ = random_split(dataset=dataset, lengths=[0.000001, 0, 0.999999])
    train_set, val_set, _ = random_split(dataset=dataset, lengths=[0.000001, 0, 0.999999])

    train_loader = DataLoader(dataset=train_set, batch_size=1, shuffle=False, num_workers=19)
    val_loader = DataLoader(dataset=val_set, batch_size=1, shuffle=False, num_workers=19)
    return train_loader, val_loader

def main(): 
    pl.seed_everything(seed=42) # TODO: Move everything to a config file
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    load_dir = '~/ehrensberger/mimicgen/datasets/core'
    # tasks = ['stack_d0_depth.hdf5', 'stack_d1_depth.hdf5']
    tasks = ['stack_d0_depth.hdf5']
    
    dataset = FinetuneR3MData(load_dir=load_dir, tasks=tasks, views=['agentview_image'], use_transform=False)
    print(len(dataset))
    
    # model = AutoEncoder()
    model = AutoEncoderR3MOnly()
    model.to(device) 
    
    # Log autoencoder training statistics and save best model checkpoint
    logger = WandbLogger(
        name='r3m_only_overfit', 
        save_dir='/home/ubuntu/ehrensberger/master-thesis/master-thesis/thesis/logs/', 
        version='small_latent_l1_agentview',
        project='finetune_r3m',
        log_model=False
        )
    
    # Save the model periodically by monitoring the total validation loss of the VAE 
    checkpoint = ModelCheckpoint(
        filename='best_finetune', 
        monitor='val_loss', 
        save_top_k=1, 
        mode='min',
        every_n_epochs=1000,       
        )
    
    trainer = Trainer(
        logger=logger, 
        max_epochs=400,  
        # callbacks=[checkpoint]
        )
    
    train_loader, val_loader = get_dataloaders(dataset=dataset)
    
    # for i, _ in train_loader: 
        
    #     print(torch.min(i))
    #     print(torch.max(i))
    #     print('-----------------------')


    print(len(train_loader))
    trainer.fit(model=model, train_dataloaders=train_loader)
    # trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    
if __name__ == '__main__': 
    main()