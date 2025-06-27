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
        
        self.fc = nn.Sequential(
            nn.Linear(self.latent_size, 256), 
            nn.ReLU(), 
            nn.Linear(256, 128),
            )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 64, kernel_size=4, stride=3, padding=1, output_padding=1),  # 4->12
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1),  # 12->12
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=10, stride=2, padding=0),  # 12->32
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, kernel_size=3, stride=1, padding=1),  # 32->32
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=22, stride=2, padding=0),  # 32->84
            nn.Sigmoid()
        )

    def forward(self, z):
        x = self.fc(z) # (self.latent_size=512, 128=8*4*4)
        # print(f"max: {torch.min(x)}")
        # print(f"min {torch.max(x)}")
        # print('-------------')
        x = x.view(-1, 8, 4, 4)
        # x = x.view(-1, 64, 21, 21) # (B, 64, 21, 21)
        x = self.decoder(x) # (B, 3, 84, 84)
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
        optimizer = torch.optim.Adam(params=self.parameters(), lr=1e-4) # TODO: Think about using two (or more) optimizers for the respective models
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)

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
        train_loss = nn.functional.l1(input=x_hat, target=x)
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

# No downsampling necessary as we directly input the color images to the r3m network 
class AutoEncoderR3MOnly(pl.LightningModule): 
    def __init__(self):
        super().__init__()
        self.model = r3m.load_r3m("resnet18") # resnet34, resnet50
        self.model.eval() 
        
        for param in self.model.parameters(): 
            param.requires_grad = False 
        
        self.upsample = Upsample(latent_size=512)
        
        # for name, module in self.model.named_modules(): 
        #     if isinstance(module, nn.BatchNorm2d): 
        #         if 'layer4.1' in name:
        #             module.train() 
        #             module.reset_running_stats()
        #             module.reset_parameters()
        #         else: 
        #             module.eval()
    
        # # Freeze every but the last layer
        # for name, param in self.model.named_parameters(): 
        #     if 'layer4.1' in name: 
        #         param.requires_grad = True 
        #     else: 
        #         param.requires_grad = False 
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(params=self.parameters(), lr=1e-3) # TODO: Think about using two (or more) optimizers for the respective models
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.05)
        # scheduler = torch.optim.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
        ## return optimizer
        return {'optimizer': optimizer, 
            'lr_scheduler': {
                'scheduler': scheduler, 
                'interval': 'step'
            }
        }
        # return {'optimizer': optimizer, 
        #         'lr_scheduler': {
        #             'scheduler': scheduler, 
        #             'monitor': 'val_loss', 
        #             'interval': 'epoch', 
        #             'frequency': 1
        #         }
        # }
    
    def forward(self, x): 
        # x = self.downsample(x)
        # print(x.shape)
        # x = self.model(x)
        # print(x.shape)
        # x = self.upsample(x)
        # print(x.shape)
        return self.upsample(self.model(x))
        
    def training_step(self, batch, batch_idx):
        rgb_image, _ = batch 
        rgb_image = rgb_image / 255.0
        rgb_hat = self.forward(rgb_image)
        
        values = {
            'min_gold': torch.min(rgb_image), 
            'max_gold': torch.max(rgb_image), 
            'min_hat': torch.min(rgb_hat), 
            'max_hat': torch.max(rgb_hat), 
            } 
        
        self.log_dict(dictionary=values)
        
        train_loss = nn.functional.mse_loss(input=rgb_hat, target=rgb_image)
        self.log(name="train_loss", value=train_loss, prog_bar=True)
        
        if self.global_step%10 == 0: 
            self.logger.experiment.log({
                "reconstruction": wandb.Image(rgb_hat[0].detach().cpu().permute(1, 2, 0).numpy(), caption="Reconstructed RGB image"),
                "gold": wandb.Image(rgb_image[0].detach().cpu().permute(1, 2, 0).numpy(), caption="Original RGB image"),
                "global_step": self.global_step
            })
        return train_loss 
    
    # def validation_step(self, batch, batch_idx):
    #     x = batch
    #     x_hat = self.forward(x)
    #     val_loss = nn.functional.mse_loss(input=x_hat, target=x)
    #     self.log(name="val_loss", value=val_loss, prog_bar=True)
    #     print(x_hat.shape)
        
        # interval = 1
        # if batch_idx%interval: 
        #     print(x_hat.shape)
            # rgb_image = wandb.Image(x_hat[0, :3, ...].detach().cpu().numpy(), caption="Reconstructed RGB image")
            # self.log({"rgb": rgb_image})
            # depth_image = wandb.Image(x_hat[0, 4, ...].detach().cpu().numpy(), caption="Reconstructed depth map")