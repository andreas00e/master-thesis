import os
import argparse
import copy 
import matplotlib.pyplot as plt 

from omegaconf import OmegaConf
from pathlib import Path
from PIL import Image

import torch
torch.cuda.empty_cache()
import torchvision.transforms as T
from torch.utils.data import DataLoader, TensorDataset, Dataset
import torch.multiprocessing as mp
from torch.nn.functional import interpolate

import lightning.pytorch as pl
from lightning.pytorch.trainer import Trainer

from utils import get_timestamp, instantiate_from_config
from models.autoencoder.autoencoder import DownsampleCVAE


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

    parser.add_argument(
        '--config_name',
        default='downsample_cvae'
    )

    parser.add_argument(
        '--devices',
        type=str,
        default='0',
    )
    parser.add_argument(
        '--horizon',
        type=int,
        default=16
    )

    return parser.parse_args()

class MyDictDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        self.data[index]["image"] = interpolate(self.data[index]["image"], size=(240, 240))

        return self.data[index]

    def __len__(self):

        return len(self.data)
    


def main(): 
    mp.set_start_method("spawn", force=True)
    args = get_parser_args()
    raw_config = OmegaConf.load(f'configs/train.yaml')
    OmegaConf.resolve(raw_config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    batch_size = 10
    image = torch.from_numpy(plt.imread("parrot.jpg").copy()).permute(2, 0, 1)
    # print(image.shape)
    list = [{"action" : torch.rand([16, 7]).to(torch.float32).to(device), "image" : image.to(torch.float32).expand(16, -1, -1, -1).to(device)}] * batch_size

    data = MyDictDataset(list)

    train_loader = DataLoader(dataset=data, num_workers=1, batch_size=5)
    val_loader = DataLoader(dataset=data, num_workers=1, batch_size=5)
    print(next(iter(train_loader))["image"].shape)

    config = preprocess_config(raw_config, args)

    epoch_length = len(train_loader) // len(config.trainer.kwargs.devices)
    config.model.kwargs.training_kwargs['num_training_steps'] = epoch_length * config.trainer.kwargs.max_epochs

    pl.seed_everything(raw_config.seed)
    model: pl.LightningModule = DownsampleCVAE(config.model.kwargs.model_kwargs, config.model.kwargs.training_kwargs)
    model.to(device)
    print(f"Currently allocated GPU-memory: {torch.cuda.memory_allocated()}")
    # exit()
    trainer = Trainer()
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)

if __name__ == '__main__':
    main()