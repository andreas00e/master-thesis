import torch 
import torch.nn as nn 
import torch.nn.functional as F 

import lightning.pytorch as pl 

import wandb 

import r3m 

class Downsample(nn.Module):
    def __init__(self):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, stride=1, padding=1), # (4, 84, 84) -> (32, 84, 84)
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), # (32, 84, 84) -> (64, 84, 84)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), # (64, 84, 84) -> (64, 168, 168)
            nn.Conv2d(64, 64, kernel_size=3, padding=1), # (64, 168, 168) -> (64, 168, 168)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Upsample(size=(224, 224), mode='bilinear', align_corners=False), # (64, 168, 168) -> (64, 224, 224)
            nn.Conv2d(64, 32, kernel_size=3, padding=1), # (64, 224, 224) -> (32, 224, 224)
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 3, kernel_size=3, padding=1), # (32, 224, 224) -> (3, 224, 224)
            nn.Tanh()  # -> [-1, 1]
        )

    def forward(self, x):
        x = (self.down(x) + 1) / 2 * 255.0
        return x

class Upsample(nn.Module):
    def __init__(self, latent_size):
        super().__init__()
        self.latent_size = latent_size
        self.fc = nn.Linear(self.latent_size, 64 * 21 * 21) # (B, 64*21*21)

        self.decoder = nn.Sequential(
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1), # (B, 32, 42, 42)
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1), # (B, 16, 84, 84)
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(16, 4, kernel_size=3, stride=1, padding=1),  # (B, 4, 84, 84)
            nn.Tanh()
        )

    def forward(self, z):
        x = self.fc(z) # (B, 64*21*21)
        x = x.view(-1, 64, 21, 21) # (B, 64, 21, 21)
        x = self.decoder(x) # (B, 4, 84, 84)
        return x

class AutoEncoder(pl.LightningModule): 
    def __init__(self):
        super().__init__()
        self.downsample = Downsample() # Downsample input w.r.t. #channels and size of image 
        self.model = r3m.load_r3m("resnet18") # resnet34, resnet50
        self.upsample = Upsample(latent_size=512)
        
        for name, module in self.model.named_modules(): 
            if isinstance(module, nn.BatchNorm2d): 
                if 'layer4.1' in name:
                    module.train() 
                    module.reset_running_stats()
                    module.reset_parameters()
                else: 
                    module.eval()
    
        # Freeze every but the last layer
        for name, param in self.model.named_parameters(): 
            if 'layer4.1' in name: 
                param.requires_grad = True 
            else: 
                param.requires_grad = False 
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(params=self.parameters(), lr=1e-3) # TODO: Think about using two (or more) optimizers for the respective models
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

        return {'optimizer': optimizer, 
                'lr_scheduler': {
                    'scheduler': scheduler, 
                    'interval': 'step'
                }
        }
    
    def forward(self, x): 
        # x = self.downsample(x)
        # print(x.shape)
        # x = self.model(x)
        # print(x.shape)
        # x = self.upsample(x)
        # print(x.shape)
        return self.upsample(self.model(self.downsample(x)))
        
    def training_step(self, batch, batch_idx):
        x = batch
        x_hat = self.forward(x)
        train_loss = nn.functional.mse_loss(input=x_hat, target=x)
        self.log(name="train_loss", value=train_loss, prog_bar=True)
        return train_loss 
    
    def validation_step(self, batch, batch_idx):
        x = batch
        x_hat = self.forward(x)
        val_loss = nn.functional.mse_loss(input=x_hat, target=x)
        self.log(name="val_loss", value=val_loss, prog_bar=True)
        print(x_hat.shape)
        
        interval = 1
        if batch_idx%interval: 
            print(x_hat.shape)
            # rgb_image = wandb.Image(x_hat[0, :3, ...].detach().cpu().numpy(), caption="Reconstructed RGB image")
            # self.log({"rgb": rgb_image})
            # depth_image = wandb.Image(x_hat[0, 4, ...].detach().cpu().numpy(), caption="Reconstructed depth map")
            # self.log({"depth": depth_image})
            
            
            
class AutoEncoderMDL(pl.LightningModule): 
    def __init__(self):
        super().__init__()
        self.downsample = Downsample() # Downsample input w.r.t. #channels and size of image 
        self.model = r3m.load_r3m("resnet18") # resnet34, resnet50
        self.upsample = Upsample(latent_size=512)
        
        for name, module in self.model.named_modules(): 
            if isinstance(module, nn.BatchNorm2d): 
                if 'layer4.1' in name:
                    module.train() 
                    module.reset_running_stats()
                    module.reset_parameters()
                else: 
                    module.eval()
    
        # Freeze every but the last layer
        for name, param in self.model.named_parameters(): 
            if 'layer4.1' in name: 
                param.requires_grad = True 
            else: 
                param.requires_grad = False 
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(params=self.parameters(), lr=1e-3) # TODO: Think about using two (or more) optimizers for the respective models
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

        return {'optimizer': optimizer, 
                'lr_scheduler': {
                    'scheduler': scheduler, 
                    'interval': 'step'
                }
        }
    
    def forward(self, x): 
        # x = self.downsample(x)
        # print(x.shape)
        # x = self.model(x)
        # print(x.shape)
        # x = self.upsample(x)
        # print(x.shape)
        return self.upsample(self.model(self.downsample(x)))
        
    def training_step(self, batch, batch_idx):
        x = batch
        x_hat = self.forward(x)
        train_loss = nn.functional.mse_loss(input=x_hat, target=x)
        self.log(name="train_loss", value=train_loss, prog_bar=True)
        return train_loss 
    
    def validation_step(self, batch, batch_idx):
        x = batch
        x_hat = self.forward(x)
        val_loss = nn.functional.mse_loss(input=x_hat, target=x)
        self.log(name="val_loss", value=val_loss, prog_bar=True)
        print(x_hat.shape)
        
        interval = 1
        if batch_idx%interval: 
            print(x_hat.shape)
            # rgb_image = wandb.Image(x_hat[0, :3, ...].detach().cpu().numpy(), caption="Reconstructed RGB image")
            # self.log({"rgb": rgb_image})
            # depth_image = wandb.Image(x_hat[0, 4, ...].detach().cpu().numpy(), caption="Reconstructed depth map")
        