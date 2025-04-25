import os
import copy
from omegaconf import OmegaConf
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.optimization import get_cosine_schedule_with_warmup

import lightning.pytorch as pl

from RoLD.models.autoencoder.common import DiagonalGaussianDistribution, AutoencoderLoss
from RoLD.models.common import SinusoidalPosEmb, get_pe, WrappedTransformerEncoder, WrappedTransformerDecoder, ResBottleneck, ImageAdapter
from RoLD.utils import instantiate_from_config 

from models.common import Clip, ParametrizedSofmax, JointAttentionEncoder

class DownsampleCVAE(pl.LightningModule):
    def __init__(
        self,
        model_kwargs,
        # joint_encoder_kwargs, 
        training_kwargs,
        mode,  # pretraining, finetuning, inference
        all_config=None
    ):
        super().__init__()
        ckpt_path = model_kwargs.ckpt_path
        if ckpt_path is not None:
            ckpt = torch.load(ckpt_path, map_location='cpu')
            # reloading config from ckpt
            hyper_params = copy.deepcopy(ckpt['hyper_parameters'])

            # replace all the model config to the original
            # but keep low_dim_feature_dim obtained in main.preprocess_config
            low_dim_feature_dim = model_kwargs.low_dim_feature_dim
            model_kwargs = hyper_params['model_kwargs']
            model_kwargs.low_dim_feature_dim = low_dim_feature_dim

        # initialze model
        self.all_config = all_config
        self.training_kwargs = training_kwargs
        self.model_kwargs = model_kwargs
        # self.joint_encoder_kwargs = joint_encoder_kwargs
        self.save_hyperparameters()

        self.action_dim = action_dim = model_kwargs['action_dim']
        self.hidden_size = hidden_size = model_kwargs['hidden_size']
        self.latent_size = latent_size = model_kwargs['latent_size']
        self.horizon = horizon = model_kwargs['horizon']

        self.action_emb = nn.Linear(action_dim, hidden_size)

        self.cls = nn.Parameter(data=torch.zeros(size=(1, hidden_size)), requires_grad=True)
        self.z_encoder = WrappedTransformerEncoder(**model_kwargs)
        self.z_down = nn.Linear(hidden_size, latent_size * 2)

        self.z_up = nn.Linear(latent_size, hidden_size)
        self.conditioner = WrappedTransformerEncoder(**model_kwargs)
        self.decoder = WrappedTransformerDecoder(**model_kwargs)

        self.action_head = nn.Linear(hidden_size, action_dim)

        # self.joint_encoder = JointAttentionEncoder(**joint_encoder_kwargs)
        
        # TODO: Fix dataloading: KeyError: b'KUKA' -> 'KUKA
        self.robot_dict = {'SAWYER': torch.tensor([0, 0, 1]), 'WIDOWX': torch.tensor([0, 1, 0]), 
                           'BAXTER': torch.tensor([0, 1, 1]), 'KUKA': torch.tensor([1, 0, 0]), 'FRANKA': torch.tensor([1, 0, 1])}

        self.loss = AutoencoderLoss(
            **training_kwargs.loss_kwargs
        )

        self.register_buffer(
            'pe', get_pe(hidden_size=hidden_size, max_len=horizon*2))

        self.with_obs = model_kwargs.get('with_obs', True)
        if self.with_obs:
            if model_kwargs.get('low_dim_feature_dim') is not None:
                assert mode == 'finetuning' or mode == 'inference'
                self.low_dim_emb = nn.Linear(model_kwargs['low_dim_feature_dim'], hidden_size)
            else:
                assert mode == 'pretraining'
                self.low_dim_emb = None

        if self.with_obs:  # fix this in config
            if hidden_size == 512:
                self.img_emb = ResBottleneck(hidden_size=hidden_size)
            else:
                self.img_emb = ImageAdapter(in_dim=512, out_dim=hidden_size)

        self.with_language = model_kwargs.get('with_language', False)
        if self.with_language:
            self.language_emb = nn.Linear(in_features=768, out_features=hidden_size)

        self.last_training_batch = None

        if ckpt_path is not None:
            if mode == 'finetuning':
                self.load_state_dict(ckpt['state_dict'], strict=False)  # no low_dim during pretraining
            elif mode == 'inference' or mode == 'pretraining':  # pretraining ldm load the pretrained ae
                self.load_state_dict(ckpt['state_dict'])  # load the whole ckpt
            del ckpt
            print(f'WARNING: ignoring AE config, AE loaded from {ckpt_path}')
        else:
            self.apply(self._init_weights)
    
    def _init_weights(self, module):
        ignore_types = (nn.Dropout, 
            SinusoidalPosEmb, 
            nn.TransformerEncoderLayer, 
            nn.TransformerDecoderLayer,
            nn.TransformerEncoder,
            nn.TransformerDecoder,
            nn.ModuleList,
            nn.Mish,
            nn.Sequential,
            WrappedTransformerDecoder,
            WrappedTransformerEncoder,
            JointAttentionEncoder,
            nn.LeakyReLU,
            nn.ELU, 
            nn.Tanh, 
            Clip,
            ParametrizedSofmax,
            ResBottleneck,
            AutoencoderLoss,
            ImageAdapter
        )
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.MultiheadAttention):
            weight_names = [
                'in_proj_weight', 'q_proj_weight', 'k_proj_weight', 'v_proj_weight']
            for name in weight_names:
                weight = getattr(module, name)
                if weight is not None:
                    torch.nn.init.normal_(weight, mean=0.0, std=0.02)
            
            bias_names = ['in_proj_bias', 'bias_k', 'bias_v']
            for name in bias_names:
                bias = getattr(module, name)
                if bias is not None:
                    torch.nn.init.zeros_(bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
        elif isinstance(module, DownsampleCVAE):
            torch.nn.init.normal_(module.cls, mean=0.0, std=0.02)
        elif isinstance(module, ignore_types):
            # no param
            pass
        else:
            raise RuntimeError("Unaccounted module {}".format(module))
    
    def configure_optimizers(self):
        kwargs = self.training_kwargs
        tuned_parameters = [p for p in self.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(
            tuned_parameters,
            lr=kwargs.lr,
        )
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=kwargs.warmup_steps, num_training_steps=kwargs.num_training_steps)

        self.lr_scheduler = scheduler
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step'
            }
        }
    
    def get_obs_emb(self, raw_image_features, raw_low_dim_data):
        if self.with_obs:
            image_emb = self.img_emb(raw_image_features)
            if raw_low_dim_data is not None and self.low_dim_emb is not None:
                low_dim_emb = self.low_dim_emb(raw_low_dim_data)
                return torch.cat([image_emb, low_dim_emb], dim=1)
            else:
                return image_emb
        else:
            return None

    def get_language_emb(self, raw_language_features):
        if self.with_language:
            return self.language_emb(raw_language_features)
        else:
            return None
    
    def encode(self, batch):
        # TODO: change 'actions' to 'action' in dataloading to stay closer to original repo implementation 
        action = batch['actions']
        image  = batch['image']
        # 2025-03-10: added JoinAttentionEncoder from: TODO: insert paper name here
        # qpos = batch['qpos'].to(torch.float)
        # qvel = batch['qvel'].to(torch.float)
        # state = batch['states'].to(torch.float)

        batch_size = action.shape[0]
        # print(action.shape)
        # print(state.shape) 
        action = action.squeeze(-1)
        # 2025-03-30: Get rid of gripper as there are still some infinte values 
        # state = state[:, :-1]
        # action = action.unsqueeze(1).repeat(1, self.horizon, 1)
        # print(f"shape action {action.shape}")

        # joint_state = torch.cat([qpos, qvel], dim=-1).to(torch.float).to(self.device) # -> Linear network:
        # fixed-size vector uniquely describing the joint by using characteristic properties 
        # joint_desc = torch.stack([self.robot_dict['KUKA'] for _ in range(joint_state.shape[1])]).expand((batch_size, -1, -1)).to(torch.float).to(self.device)# -> torch.Size([14, 3]) -> torch.Size([8, 14, 3])
        # joint_emb = self.joint_encoder(joint_desc, joint_state, state)
        # print(joint_emb.shape)
        # changed on 2025-03-07: no raw_low_dim_data in dataset; incompatible tensor shapes between batch['iamge'] and weight matrix
        obs_emb = self.get_obs_emb(raw_image_features=image.view(batch_size, -1, self.hidden_size), raw_low_dim_data=None)
        
        # print(f"pe shape: {self.pe[:, :self.horizon, :].expand((batch_size, self.horizon, self.hidden_size)).shape}")
        pos_action_emb = self.action_emb(action) + self.pe[:, :self.horizon, :].expand((batch_size, self.horizon, self.hidden_size))
        cls = self.cls.expand((batch_size, 1, self.hidden_size))

        z_encoder_input = torch.cat([cls, pos_action_emb], dim=1)
        if obs_emb is not None:
            z_encoder_input = torch.cat([z_encoder_input, obs_emb], dim=1)

        z_encoder_output = self.z_encoder(z_encoder_input)[:, 0:1, :]
        z_encoder_output = self.z_down(z_encoder_output)
        posterior = DiagonalGaussianDistribution(z_encoder_output)
        return posterior, z_encoder_output, obs_emb
    
    def decode(self, obs_emb, posterior=None, z=None, sample_posterior=True, raw_language_features=None):
        if z is None:
            if sample_posterior:
                z = posterior.sample()
            else:
                z = posterior.mode()
        z = self.z_up(z)
        batch_size = z.shape[0]
        
        condition_input = z
        if obs_emb is not None:
            condition_input = torch.cat([obs_emb, condition_input], dim=1)  # obs_emb, z
        if self.with_language:
            condition_input = torch.cat([self.get_language_emb(raw_language_features), condition_input], dim=1)  # lang, obs, z
        condition = self.conditioner(condition_input)

        decoder_input = self.pe[:, :self.horizon, :].expand((batch_size, self.horizon, self.hidden_size))
        decoder_output = self.decoder(tgt=decoder_input, memory=condition)
        pred_action = self.action_head(decoder_output)
        return pred_action

    def forward(self, batch, batch_idx, sample_posterior=True, split='train'):
        posterior, obs_emb, _ = self.encode(batch)
        # changed on 2025-03-10: Currently no available language features 
        # pred_action = self.decode(posterior=posterior, obs_emb=obs_emb, sample_posterior=sample_posterior, raw_language_features=batch['language'])
        pred_action = self.decode(posterior=posterior, obs_emb=obs_emb, sample_posterior=sample_posterior, raw_language_features=None)

        total_loss, log_dict = self.loss.recon_kl_loss(
            inputs=batch['actions'].squeeze(-1).to(torch.float), reconstructions=pred_action, posteriors=posterior, split=split)
        return total_loss, log_dict
    
    def training_step(self, batch, batch_idx):
        self.last_training_batch = batch
        total_loss, log_dict = self.forward(batch=batch, batch_idx=batch_idx, split='train')
        self.log_dict(log_dict, prog_bar=True, sync_dist=True)
        return total_loss

    def validation_step(self, batch, batch_idx):
        total_loss, log_dict = self.forward(batch=batch, batch_idx=batch_idx, split='val')
        self.log_dict(log_dict, prog_bar=True, sync_dist=True)
        return total_loss
    
    def test_step(self, batch, batch_idx):
        total_loss, log_dict = self.forward(batch=batch, batch_idx=batch_idx, split='test')
        self.log_dict(log_dict, prog_bar=True, sync_dist=True)
        return total_loss
    
    def predict_step(self, batch, batch_idx, sample_posterior=True):
        # list 
        robots = batch['robot']
        # TODO: Implement true prediction step (used for infering actions, not visualizing the laten space) 
        _, _, z_encoder_output = self.encode(batch)
        # latent vector (mean of distribution)
        mean, _ = torch.chunk(z_encoder_output, chunks=2, dim=-1)
        mean = mean[:, 0, :].squeeze(1).cpu().numpy() 
        result = list(zip(robots, mean))
        return result 

