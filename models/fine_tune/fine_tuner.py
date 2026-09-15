from omegaconf import DictConfig
from hydra.utils import instantiate
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn 
import torch.nn.functional as F 
import lightning.pytorch as pl
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

from models.utils.vicreg import VICReg
from models.utils.vision import VisionEncoder, Expander


class FineTunerGripper(pl.LightningModule): 
    def __init__(
        self, 
        d_model: int, 
        fine_tuner_visual_ckpt: str,
        optimizer_kwargs: DictConfig, 
        lr_scheduler_kwargs: DictConfig 
        ) -> None:
        
        super().__init__()
        self.save_hyperparameters() 
      
        self.d_model = d_model
        self.fine_tuner_visual_ckpt = fine_tuner_visual_ckpt

        self.optimizer_kwargs = optimizer_kwargs
        self.lr_scheduler_kwargs = lr_scheduler_kwargs
        
        self.fineTunerVisual = FineTunerVisual.load_from_checkpoint(fine_tuner_visual_ckpt) 
        self.fineTunerVisual.eval() 
        self.fineTunerVisual.freeze/( )
        
        self.gripperEncoder = nn.Sequential(
            nn.Linear(1, self.d_model // 2), 
            nn.ReLU(), 
            nn.Linear(self.d_model // 2, self.d_model)
            )
    
    def setup(self, stage: str) -> None: 
        if stage == "predict" and hasattr(self, "fineTunerVisual"): 
            del self.fineTunerVisual
            torch.cuda.empty_cache() 
        
    def configure_optimizers(self) -> Dict[str, Any]:
        trainable_parameters = filter(lambda p: p.requires_grad, self.gripperEncoder.parameters())        
        optimizer = instantiate(self.optimizer_kwargs, params=trainable_parameters)
        
        if hasattr(self, "trainer") and self.trainer is not None: 
            total_steps = self.trainer.estimated_stepping_batches 
            
            warmup_percentage = self.lr_scheduler_kwargs.get("warmup_percentage", 0.1)
            warmup_steps = int(warmup_percentage * total_steps)
            decay_steps = total_steps - warmup_steps
            
            self.lr_scheduler_kwargs.linear.total_iters = warmup_steps 
            self.lr_scheduler_kwargs.cosine_annealing.T_max = decay_steps 

        scheduler_one = LinearLR(optimizer, **self.lr_scheduler_kwargs.linear)
        scheduler_two = CosineAnnealingLR(optimizer, **self.lr_scheduler_kwargs.cosine_annealing)

        scheduler = SequentialLR(optimizer, schedulers=[scheduler_one, scheduler_two],  milestones=[warmup_steps])
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler, 
                "interval": "step", 
                "frequency": 1
            }
        }
        
    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor: 
        return self(batch, batch_idx, stage="train")
    
    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor: 
        return self(batch, batch_idx, stage="val")
    
    def test_step(self, batch: Any, batch_idx: int) -> torch.Tensor: 
        return self(batch, batch_idx, stage="test")
    
    def predict_step(self, batch: Any, batch_idx: int) -> torch.Tensor: 
        return self.gripperEncoder(batch["g_qpos"])

    def forward(self, batch: Any, batch_idx: int, stage: str) -> torch.Tensor: 
        g_qpos = self.gripperEncoder(batch["g_qpos"])
        
        with torch.no_grad():  
            rgb_one_y = self.fineTunerVisual.VisionEncoder(batch["rgb_one"])
        
        loss = F.mse_loss(g_qpos, rgb_one_y)
        
        self.log_dict(
            {
                f"{stage}/loss": loss, 
            }, 
            logger=True,
            prog_bar=True, 
            on_step=stage == "train", 
            on_epoch=True, 
            sync_dist=True, 
        )
        
        return loss
        

class FineTunerVisual(pl.LightningModule): 
    def __init__(
        self, 
        optimizer_kwargs: DictConfig, 
        lr_scheduler_kwargs: DictConfig, 
        vision_encoder_kwargs: DictConfig, 
        expander_kwargs: DictConfig, 
        vic_reg_kwargs: DictConfig,
        ) -> None: 
        
        super().__init__()
        self.save_hyperparameters() 
                        
        self.optimizer_kwargs = optimizer_kwargs
        self.lr_scheduler_kwargs = lr_scheduler_kwargs
        self.vision_encoder_kwargs = vision_encoder_kwargs 
        self.expander_kwargs = expander_kwargs
        self.vic_reg_kwargs = vic_reg_kwargs
        
        self.visionEncoder = VisionEncoder(**self.vision_encoder_kwargs)
        self.visionExpander = Expander(**self.expander_kwargs)
        self.vicReg = VICReg(logger=None, **self.vic_reg_kwargs)
                
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        
    def setup(self, stage: Optional[str]=None) -> None:
        if self.logger is not None: 
            self.vicReg.logger = self.logger 
            
    def configure_optimizers(self) -> Dict[str, Any]:
        trainable_parameters = filter(lambda p: p.requires_grad, self.parameters())        
        optimizer = instantiate(self.optimizer_kwargs, params=trainable_parameters)
        
        if hasattr(self, "trainer") and self.trainer is not None: 
            total_steps = self.trainer.estimated_stepping_batches 
            
            warmup_percentage = self.lr_scheduler_kwargs.get("warmup_percentage", 0.1)
            warmup_steps = int(warmup_percentage * total_steps)
            decay_steps = total_steps - warmup_steps
            
            self.lr_scheduler_kwargs.linear.total_iters = warmup_steps 
            self.lr_scheduler_kwargs.cosine_annealing.T_max = decay_steps 

        scheduler_one = LinearLR(optimizer, **self.lr_scheduler_kwargs.linear)
        scheduler_two = CosineAnnealingLR(optimizer, **self.lr_scheduler_kwargs.cosine_annealing)

        scheduler = SequentialLR(optimizer, schedulers=[scheduler_one, scheduler_two],  milestones=[warmup_steps])
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler, 
                "interval": "step", 
                "frequency": 1
            }
        }
    
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None: 
        trainable_param_names = {name for name, param in self.named_parameters() if param.requires_grad}
        filtered_state_dict = {key: val for key, val in checkpoint["state_dict"].items() if key in trainable_param_names}

        checkpoint["state_dict"] = filtered_state_dict
    
    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        self.strict_loading = False
        
    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        return self(batch, batch_idx, stage="train")
    
    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        return self(batch, batch_idx, stage="val")
    
    def test_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        rgb_one = batch["rgb_one"]
        rgb_two = batch["rgb_two"]
        
        rgb_one_ours = self.visionEncoder(rgb_one) # [n, d_model] robot0_eye_in_hand_image 
        rgb_two_ours = self.visionEncoder(rgb_two) # [n, d_model] agentview_image 

        rgb_one_r3m = self.visionEncoder.backbone(rgb_one) 
        rgb_two_r3m = self.visionEncoder.backbone(rgb_one)
        
        rgb_one_ours = F.normalize(rgb_one_ours)
        rgb_two_ours = F.normalize(rgb_two_ours)
        
        rgb_one_r3m = F.normalize(rgb_one_r3m)
        rgb_two_r3m = F.normalize(rgb_two_r3m)

        sim_ours = self.cos(rgb_one_ours, rgb_two_ours)
        sim_r3m = self.cos(rgb_one_r3m, rgb_two_r3m)
        sim_diff = sim_ours - sim_r3m
        
        out = {
            "sim_diff": sim_diff.detach().cpu(), 
        }
        
        return out 
    
    def predict_step(self, batch: Any, batch_idx: int) -> Tuple[torch.Tensor, torch.Tensor]: 
        rgb_one_y = self.visionEncoder(batch["rgb_one"]) # [n, d_model] robot0_eye_in_hand_image 
        rgb_two_y = self.visionEncoder(batch["rgb_two"]) # [n, d_model] agentview_image 
        
        return rgb_one_y, rgb_two_y
              
    def check_bn_stats(self, model, batch, layer_names=None):
        results = {}
        hooks = []

        def make_hook(name):
            def hook(module, input, output):
                x = input[0]
                actual_mean = x.mean(dim=(0, 2, 3))
                actual_var = x.var(dim=(0, 2, 3), unbiased=False)
                mean_diff = (actual_mean - module.running_mean).abs()
                var_ratio = actual_var / (module.running_var + 1e-8)
                results[name] = {
                    "mean_abs_diff_max": mean_diff.max().item(),
                    "mean_abs_diff_mean": mean_diff.mean().item(),
                    "var_ratio_min": var_ratio.min().item(),
                    "var_ratio_max": var_ratio.max().item(),
                    "var_ratio_mean": var_ratio.mean().item(),
                }
            return hook

        for name, module in model.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                if layer_names is None or any(n in name for n in layer_names):
                    hooks.append(module.register_forward_hook(make_hook(name)))

        was_training = model.training
        model.eval()
        with torch.no_grad():
            model(batch)
        model.train(was_training)

        for h in hooks:
            h.remove()
        return results
    
    def on_train_batch_end(self, outputs, batch, batch_idx):
        if batch_idx % 100 == 0:
            stats = self.check_bn_stats(self.visionEncoder, batch["rgb_two"])
            for name, s in stats.items():
                mean_diff_ratio = s['mean_abs_diff_max'] / s['mean_abs_diff_mean']
                var_ratio = s['var_ratio_min'] / s['var_ratio_max']

                self.log_dict({
                    f"{name}/mean_diff_ratio": mean_diff_ratio,
                    f"{name}/var_ratio": var_ratio,
                }, on_step=True, on_epoch=False)
    
    def on_test_batch_end(self, outputs, batch, batch_idx): 
        return None
    
    def forward(self, batch: Any, batch_idx: int, stage: str) -> torch.Tensor:       
        rgb_one_y = self.visionEncoder(batch["rgb_one"]) # [n, d_model] robot0_eye_in_hand_image 
        rgb_two_y = self.visionEncoder(batch["rgb_two"]) # [n, d_model] agentview_image 
        
        rgb_one_z = self.visionExpander(rgb_one_y) # [n, d_model*4]
        rgb_two_z = self.visionExpander(rgb_two_y) # [n, d_model*4]

        loss, logs_ = self.vicReg(rgb_one_z, rgb_two_z)
        
        self.log_dict(
            {
                f"{stage}/inv_loss": logs_["inv_loss"], 
                f"{stage}/var_loss": logs_["var_loss"],
                f"{stage}/cov_loss": logs_["cov_loss"],
                f"{stage}/tot_loss": loss
            }, 
            logger=True,
            prog_bar=True, 
            on_step=stage == "train", 
            on_epoch=True, 
            sync_dist=True, 
        )

        return loss 