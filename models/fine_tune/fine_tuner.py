from r3m import load_r3m 
from omegaconf import DictConfig
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn 
import torch.nn.functional as F 
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

import lightning.pytorch as pl

from models.utils.vicreg import VICReg
from models.utils.aux_models import VisionEncoder, Expander
from models.fine_tune.tests import _pair_metrics
        

class FineTunerVisual(pl.LightningModule): 
    def __init__(
        self, 
        optimizer_kwargs: DictConfig, 
        scheduler_kwargs: DictConfig, 
        vision_encoder_kwargs: DictConfig, 
        expander_kwargs: DictConfig, 
        vic_reg_kwargs: DictConfig,
        ) -> None: 
        
        super().__init__()
        self.save_hyperparameters() 
                        
        self.optimizer_kwargs = optimizer_kwargs
        self.scheduler_kwargs = scheduler_kwargs
        self.vision_encoder_kwargs = vision_encoder_kwargs 
        self.expander_kwargs = expander_kwargs
        self.vic_reg_kwargs = vic_reg_kwargs
        
        self.visionEncoder = VisionEncoder(**self.vision_encoder_kwargs)
        self.visionExpander = Expander(**self.expander_kwargs)
        self.vicReg = VICReg(logger=None, **self.vic_reg_kwargs)
        self.r3m_baseline = None # only needed for testing 
        
        self.strict_loading = False        
        
    def setup(self, stage: Optional[str]=None) -> None:
        if self.logger is not None: 
            self.vicReg.logger = self.logger 

    def _configure_scheduler(
        self, 
        optimizer: Any, # torch.optim.Optimizer 
        total_steps: int,
        ) -> Any: # torch.optim.lr_scheduler.LRScheduler
        
        warmup_percentage = self.scheduler_kwargs.get("warmup_percentage", 0.1)
        warmup_steps = int(warmup_percentage * total_steps)
        decay_steps = total_steps - warmup_steps
        
        self.scheduler_kwargs.linear.total_iters = warmup_steps
        self.scheduler_kwargs.cosine_annealing.T_max = decay_steps 
        
        scheduler_one = LinearLR(optimizer, **self.scheduler_kwargs.linear)
        scheduler_two = CosineAnnealingLR(optimizer, **self.scheduler_kwargs.cosine_annealing)
        scheduler = SequentialLR(optimizer, schedulers=[scheduler_one, scheduler_two],  milestones=[warmup_steps])
        
        return scheduler
                   
    def configure_optimizers(self):
        encoder_parameters = list(filter(lambda p: p.requires_grad, self.visionEncoder.parameters()))
        expander_parameters = list(filter(lambda p: p.requires_grad, self.visionExpander.parameters())) 
        self.visionEncoder.model.convnet.fc.weight.requires_grad = True
        self.visionEncoder.model.convnet.fc.bias.requires_grad = True

        encoder_parameters.extend([self.visionEncoder.model.convnet.fc.weight, self.visionEncoder.model.convnet.fc.bias])
         
        param_groups = [
            {"params": encoder_parameters, **self.optimizer_kwargs.encoder},
            {"params": expander_parameters, **self.optimizer_kwargs.expander}, 
        ]

        optimizer = AdamW(params=param_groups)
        total_steps = self.trainer.estimated_stepping_batches if self.trainer else 1000

        scheduler = {
            "scheduler": self._configure_scheduler(optimizer, total_steps),
            "interval": "step",
            "frequency": 1,
        }

        return {
            "optimizer": optimizer, 
            "lr_scheduler": scheduler
            }

    def on_test_start(self) -> None: 
        model_name = self.vision_encoder_kwargs.get("model_name", "resnet18")
        r3m_baseline = load_r3m(model_name).module
        self.r3m_baseline  = r3m_baseline.to(self.device).eval().requires_grad_(False)
        self.visionEncoder.eval().requires_grad_(False)
        
        # if self.trainer.datamodule is not None:
        #     self.trainer.datamodule.setup(stage="fit")
        #     train_loader = self.trainer.datamodule.train_dataloader()
            
        #     print("We are using train")
        #     # 2. Extract a single training batch
        #     train_batch = next(iter(train_loader))

        #     # 3. Move batch tensors to the model's device (e.g., GPU/MPS)
        #     if isinstance(train_batch, (list, tuple)):
        #         self.train_batch = [
        #             t.to(self.device) if isinstance(t, torch.Tensor) else t
        #             for t in train_batch
        #         ]
        #     elif isinstance(train_batch, dict):
        #         self.train_batch = {
        #             k: v.to(self.device) if isinstance(v, torch.Tensor) else v
        #             for k, v in train_batch.items()
        #         }
        #     else:
        #         self.train_batch = train_batch.to(self.device)
    
    def on_predict_start(self) -> None:
        self.visionEncoder.model.merge_adapter()

    def on_predict_end(self) -> None:
        self.visionEncoder.model.unmerge_adapter()
    
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None: 
        trainable_param_names = {name for name, param in self.named_parameters() if param.requires_grad}
        filtered_state_dict = {key: val for key, val in checkpoint["state_dict"].items() if key in trainable_param_names}

        checkpoint["state_dict"] = filtered_state_dict
    
    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        assert any("convnet.fc.weight" in k for k in checkpoint["state_dict"]), "fc missing from checkpoint"
        
    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        return self(batch, batch_idx, stage="train")
    
    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        return self(batch, batch_idx, stage="val")
    
    def test_step(self, batch: Any, batch_idx: int) -> None:
        with torch.no_grad(): 
            rgb_one = batch["rgb_one"] # [batch_size, chunk, window, channels, height, width]
            rgb_two = batch["rgb_one"] 
            
            rgb_one_ours = self.visionEncoder(rgb_one) # [n, d_model]
            rgb_two_ours = self.visionEncoder(rgb_two) # [n, d_model]
            
            for name, children in self.visionEncoder.named_children(): 
                print(name)
            exit()
            pos_our, neg_ours, top_one_ours = _pair_metrics(rgb_one_ours, rgb_two_ours)
            
            rgb_one_baseline = self.r3m_baseline(rgb_one.view(-1, *rgb_one.shape[-3:])) # [n, d_model]
            rgb_two_baseline = self.r3m_baseline(rgb_two.view(-1, *rgb_one.shape[-3:])) # [n, d_model]
            pos_basline, neg_baseline, top_one_baseline = _pair_metrics(rgb_one_baseline, rgb_two_baseline)
            
            z_one_ours = self.visionExpander(rgb_one_ours)
            z_two_ours = self.visionExpander(rgb_two_ours)
            pos_z, neg_z, top1_z = _pair_metrics(z_one_ours, z_two_ours)
        
            sim_diff_ours = pos_our.mean() - neg_ours.mean()
            sim_diff_ours_expander = pos_z.mean() - neg_z.mean()
            sim_diff_baseline = pos_basline.mean() - neg_baseline.mean()
            top_one_ours = top_one_ours.mean()
            top_one_expander = top1_z.mean()
            top_one_baseline = top_one_baseline.mean()

        self.log_dict(
            {
            "test/sim_diff_ours": sim_diff_ours,
            "test/sim_diff_ours_expander": sim_diff_ours_expander,
            "test/sim_diff_baseline": sim_diff_baseline,
            "test/top_one_ours": top_one_ours,
            "test/top_one_expander": top_one_expander,
            "test/top_one_baseline": top_one_baseline
            },
            logger=True, 
            prog_bar=True, 
            on_step=False, 
            on_epoch=True
        )

    def predict_step(self, batch: Any, batch_idx: int) -> Tuple[torch.Tensor, torch.Tensor]: 
        rgb_one_y = self.visionEncoder(batch["rgb_one"]) # [n, d_model] robot0_eye_in_hand_image 
        rgb_two_y = self.visionEncoder(batch["rgb_two"]) # [n, d_model] agentview_image 
        
        return rgb_one_y, rgb_two_y
    
    def forward(self, batch: Any, batch_idx: int, stage: str) -> torch.Tensor:  
        rgb_one_y = self.visionEncoder(batch["rgb_one"]) # [n, d_model] robot0_eye_in_hand_image 
        rgb_two_y = self.visionEncoder(batch["rgb_two"]) # [n, d_model] agentview_image 
        
        if stage == "train" and self.logger is not None and self.global_step % 50 == 0: 
            with torch.no_grad(): 
                y_centered = rgb_one_y - torch.mean(rgb_one_y, dim=0)
                
                _, S, _ = torch.linalg.svd(y_centered)
                p = S / torch.sum(S)
                eff_rank = torch.exp(-torch.sum(p*torch.log(p + 1e-12))).item() # effective Rank (Shannon entropy of a singular value)
                
                self.log(
                    "train/eff_rank", 
                    eff_rank, 
                    on_step=True, 
                    on_epoch=False,
                    sync_dist=False
                    )
            
        rgb_one_z = self.visionExpander(rgb_one_y) # [n, d_model*x]
        rgb_two_z = self.visionExpander(rgb_two_y) # [n, d_model*x]

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
            on_step=stage=="train", 
            on_epoch=True, 
            sync_dist=True, 
        )

        return loss 