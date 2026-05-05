import copy
from termcolor import colored

import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from diffusers.optimization import get_cosine_schedule_with_warmup

import lightning.pytorch as pl

from RoLD.models.common import SinusoidalPosEmb, get_pe, WrappedTransformerEncoder, WrappedTransformerDecoder, ResBottleneck

from models.vision.vision import VisionCombiner
from models.autoencoder.common import DiagonalGaussianDistribution, AutoencoderLoss


class DownsampleCVAE(pl.LightningModule):
    def __init__(
        self,
        model_kwargs,
        training_kwargs,
        mode, # pretraining, finetuning, inference  # XXX: pretraining is not avaialable! 
        all_config=None
    ):
        super().__init__()
        self.save_hyperparameters()

        
        ckpt_path = model_kwargs.ckpt_path
        if ckpt_path is not None:
            ckpt = torch.load(ckpt_path, map_location="cpu")
            hyper_params = copy.deepcopy(ckpt["hyper_parameters"])
            
            low_dim_feature_dim = model_kwargs.low_dim_feature_dim
            model_kwargs = hyper_params["model_kwargs"]
            model_kwargs.low_dim_feature_dim = low_dim_feature_dim

        self.all_config = all_config # XXX: Why do we need that if its default is None?
        self.training_kwargs = training_kwargs
        self.model_kwargs = model_kwargs

        self.action_dim = action_dim = model_kwargs["action_dim"]
        self.hidden_size = hidden_size = model_kwargs["hidden_size"]
        self.latent_size = latent_size = model_kwargs["latent_size"]
        self.horizon = horizon = model_kwargs["horizon"]

        self.action_emb = nn.Linear(action_dim, hidden_size)
        # self.lang_down = nn.Linear(hidden_size, int(hidden_size/2)) # XXX: Why? 

        self.cls = nn.Parameter(data=torch.zeros(size=(1, hidden_size)), requires_grad=True)
        self.z_encoder = WrappedTransformerEncoder(**model_kwargs)
        self.z_down = nn.Linear(hidden_size, latent_size * 2) # mu and sigma of normal distribution 
        self.z_up = nn.Linear(latent_size, hidden_size)
        self.conditioner = WrappedTransformerEncoder(**model_kwargs)
        self.decoder = WrappedTransformerDecoder(**model_kwargs)

        self.action_head = nn.Linear(hidden_size, action_dim)

        self.loss = AutoencoderLoss(
            **training_kwargs.loss_kwargs
        )

        self.register_buffer(
            "pe", get_pe(hidden_size=hidden_size, max_len=horizon*2))

        self.with_obs = model_kwargs.get("with_obs", True)
        if self.with_obs:
            if model_kwargs.get("low_dim_feature_dim") is not None:
                assert mode == "finetuning" or mode == "inference"
                self.low_dim_emb = nn.Linear(model_kwargs["low_dim_feature_dim"], hidden_size)
            else:
                assert mode == "pretraining"
                self.low_dim_emb = None
                
        if self.model_kwargs.image_model:
            img_emb_type, img_emb_version = self.model_kwargs.image_model.split(".")
            print(colored(f"Using {img_emb_type} with version {img_emb_version} as image encoder", "green"))
            self.img_emb = VisionCombiner(hidden_size=self.hidden_size, out_size=128, resnet_version=img_emb_version)
        else: 
            raise NotImplementedError(colored("No model for embedding scene images could be found!", "red"))
        
        self.img_emb.eval()
        for module in self.img_emb.modules(): 
            module.skip_init = True # skip modules during weight initialization 
            if not isinstance(module, nn.Linear): 
                for param in module.parameters(): 
                    param.requires_grad = False # image embedding module is frozen 
            elif isinstance(module, nn.Linear): 
                self.img_emb._init_weights(module)
                for param in module.parameters(): 
                    param.requires_grad = True # set parameters of nn.Linear() for down projection to 'train' # XXX: Why? 
        
        self.with_language = model_kwargs.get("with_language", False)
        
        # LANGUAGE EMBEDDING MODULE 
        if self.model_kwargs.language_model:
            lang_emb_type, lang_emb_version = self.model_kwargs.language_model.split(".")
            print(colored(f"Using {lang_emb_type} with version {lang_emb_version} as language encoder", "green"))
            if lang_emb_type == "bert": # load bert tokenizer and model 
                self.lang_tokenizer = BertTokenizer.from_pretrained(lang_emb_version) # pre-trained bert tokenizer
                self.lang_emb_model = BertModel.from_pretrained(lang_emb_version) # pre-trained bert model
                self.lang_emb = nn.Linear(in_features=768, out_features=hidden_size) # TODO: Move in_features to config 
            # elif lang_emb_type == 'clip': # load clip tokenizer and model 
            #     self.img_emb = r3m.load_r3m(img_emb_version)
            self.lang_emb_model.to(self.device)
            self.lang_emb_model.eval()
            for module in self.lang_emb_model.modules(): 
                module.skip_init = True
            for parameters in self.lang_emb_model.parameters(): 
                parameters.requires_grad = False # langugage embedding module is frozen
        else: 
            raise NotImplementedError(colored("No model for embedding language task instructions could be found!", "red"))

        self.last_training_batch = None

        if ckpt_path is not None:
            if mode == "finetuning": # XXX: The only mode for training we have as we do not have access to the "Big dataset"
                self.load_state_dict(ckpt["state_dict"], strict=False)  # no low_dim during pretraining
            elif mode == 'inference' or mode == "pretraining":  # pretraining ldm load the pretrained ae
                self.load_state_dict(ckpt["state_dict"])  # load the whole ckpt
            del ckpt
            print(f"WARNING: ignoring AE config, AE loaded from {ckpt_path}")
        else:
            self.apply(self._init_weights)
    
    def _init_weights(self, module):
        ignore_types = (
            nn.Dropout, 
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
            nn.LeakyReLU,
            AutoencoderLoss
        )
        if getattr(module, "skip_init", False): 
            return 
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.MultiheadAttention):
            weight_names = [
                "in_proj_weight", "q_proj_weight", "k_proj_weight", "v_proj_weight"]
            for name in weight_names:
                weight = getattr(module, name)
                if weight is not None:
                    torch.nn.init.normal_(weight, mean=0.0, std=0.02)
            bias_names = ["in_proj_bias", "bias_k", "bias_v"]
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
            "optimizer": optimizer, 
            "lr_scheduler": 
                {"scheduler": scheduler, 
                 "interval": "step"}
                }
    
    def get_obs_emb(self, raw_image_features, raw_low_dim_data): # XXX: What is raw_low_dim_data? 
        if self.with_obs:
            with torch.no_grad(): 
                image_emb = self.img_emb(raw_image_features) # raw_image_feature: [B, C, W, H]
            if raw_low_dim_data is not None and self.low_dim_emb is not None:
                low_dim_emb = self.low_dim_emb(raw_low_dim_data)
                return torch.cat([image_emb, low_dim_emb], dim=1)
            else:
                return image_emb
        else:
            return None
        
    def get_language_emb(self, raw_language_features):
        if self.with_language:
            tokens = self.lang_tokenizer(text=raw_language_features, padding=True, return_tensors="pt").to(self.device)
            with torch.no_grad():
                language_features = self.lang_emb_model(**tokens) 
                language_features = self.lang_emb(language_features.last_hidden_state[:, 0, :]) # pass cls token to linear layer for shape allignment 
            return language_features.unsqueeze(1)
        else:
            return None
    
    def encode(self, batch):
        actions = batch["actions"] # [B, action_horizon, action_dim]
        image = batch["image"] # [B, image_horizon, C, H, W]
        batch_size = actions.shape[0] 
        
        obs_emb = self.img_emb(image) # [B, image_horizon, hidden_size]    
        pos_action_emb = self.action_emb(actions) + self.pe[:, :self.horizon, :].expand((batch_size, self.horizon, self.hidden_size))
        cls = self.cls.expand((batch_size, 1, self.hidden_size))

        z_encoder_input = torch.cat([cls, pos_action_emb], dim=1)
        if obs_emb is not None:
            z_encoder_input = torch.cat([z_encoder_input, obs_emb], dim=1)

        z_encoder_output = self.z_encoder(z_encoder_input)[:, :1, :]
        z_encoder_output = self.z_down(z_encoder_output) # [B, 1, latent_size * 2]
        
        posterior = DiagonalGaussianDistribution(z_encoder_output)
        return posterior, obs_emb
    
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
            condition_input = torch.cat([obs_emb, condition_input], dim=1)
        if self.with_language:
            language_emb = self.get_language_emb(raw_language_features)
            condition_input = torch.cat([language_emb, condition_input], dim=1)
        condition = self.conditioner(condition_input)

        decoder_input = self.pe[:, :self.horizon, :].expand((batch_size, self.horizon, self.hidden_size))
        decoder_output = self.decoder(tgt=decoder_input, memory=condition)
        pred_action = self.action_head(decoder_output)
        return pred_action

    def forward(self, batch, batch_idx, sample_posterior=True, split="train"):
        posterior, obs_emb = self.encode(batch)
        action_hat = self.decode(posterior=posterior, obs_emb=obs_emb, sample_posterior=sample_posterior, raw_language_features=batch["text"])
        
        total_loss, log_dict = self.loss.recon_kl_loss(
            # inputs=batch['actions'].squeeze(-1).to(torch.float), reconstructions=pred_action, posteriors=posterior, kl_weight=kl_weight[self.global_step], split=split) # XXX: What? 
            inputs=batch["actions"], reconstructions=action_hat, posteriors=posterior, split=split)
        return total_loss, log_dict
    
    def training_step(self, batch, batch_idx):
        self.last_training_batch = batch
        
        total_loss, log_dict = self.forward(batch=batch, batch_idx=batch_idx, split="train")
        self.log_dict(log_dict, prog_bar=True, sync_dist=True)
        return total_loss

    def validation_step(self, batch, batch_idx):
        total_loss, log_dict = self.forward(batch=batch, batch_idx=batch_idx, split="val")
        self.log_dict(log_dict, prog_bar=True, sync_dist=True)
        return total_loss