from typing import Dict, Any
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
import wandb
from tqdm.notebook import tqdm
from .metrics import Metric
from utils.visualize import color, COLORS
import numpy as np


class Trainer:
    def __init__(self,
            config: Dict[str, Any],
            model: nn.Module,
            device: torch.device,
            train_loader: DataLoader,
            val_loader: DataLoader = None,
            class_dict: Dict[int, str] = {},
            metrics: list[Metric] = []
        ) -> None:
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.model = model.to(device)
        self.class_dict = class_dict 
        self.metrics = metrics

        # Optimization
        lr = self.config['training']['lr']
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=self.config['training']['wd']
        )

        # Learning rate schedule
        num_steps = self.config['training']['num_epochs'] * len(self.train_loader)
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=lr,
            total_steps=num_steps,
            pct_start=0.1
        )

        # AMP
        self.scaler = torch.amp.GradScaler(self.device.type)

        self.step = 0
        self.epoch = 0
        # self.path_ckpts = Path(self.config['save_dir'])
        # self.path_ckpts.mkdir(exist_ok=True)
        self.evaluation_rate = self.config['training'].get('evaluation_rate', 1)

    def log_segmentation_results(
        self,
        true: torch.Tensor,
        pred: torch.Tensor
    ) -> None:
        table = wandb.Table(columns=["Comparison"])
        
        for true_mask, pred_mask in zip(true, pred):
            true_np = true_mask.cpu().numpy()
            pred_np = pred_mask.cpu().numpy()
            
            true_colored = color(true_np, COLORS)
            pred_colored = color(pred_np, COLORS)
            
            comparison = np.concatenate((true_colored, pred_colored), axis=1)
            table.add_data(wandb.Image(comparison))
            
        wandb.log({"Segmentation Results": table})

    def train(self, run_name) -> None:
        wandb.init(
            project=self.config['wandb']['project'],
            name=run_name,
            config=self.config
        )
        wandb.define_metric("train/lr", step_metric="step")
        wandb.define_metric("train/batch_loss", step_metric="step")
        wandb.define_metric("train/epoch_loss", step_metric="epoch")
        wandb.define_metric("val/epoch_loss", step_metric="epoch")
        for metric in self.metrics:
            wandb.define_metric(f"train/{metric.__class__.__name__}", step_metric="epoch")
            wandb.define_metric(f"val/{metric.__class__.__name__}", step_metric="epoch")

        for epoch in tqdm(range(1, self.config['training']['num_epochs'] + 1), desc="Epoch", colour='green'):
            self.model.train()

            for imgs, masks in tqdm(self.train_loader, 
                                    total=len(self.train_loader), 
                                    desc=f"Epoch {epoch} Batches", 
                                    leave=False,
                                    colour='blue'):

                self._training_step(imgs, masks)

            if epoch % self.evaluation_rate == 0:
                self.eval('train', epoch)
                if self.val_loader:
                    self.eval('val', epoch)

        wandb.finish()

    def _training_step(self, imgs: torch.Tensor, masks: torch.Tensor) -> float:
        imgs = imgs.to(self.device, non_blocking=True)
        masks = masks.long().to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=self.device.type, dtype=torch.float16):
            _, logits = self.model(imgs)
            loss = F.cross_entropy(logits, masks)

        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()

        self.step += 1 
        
        if self.step % 10 == 0:
            wandb.log({
                'train/batch_loss': loss.item(),
                'train/lr': self.scheduler.get_last_lr()[0],
                'step': self.step
            })

    @torch.inference_mode()
    def eval(self, split: str, epoch: int) -> None:
        self.model.eval()
        loader = self.val_loader if split == "val" else self.train_loader

        cumulative_loss = 0.0
        for metric in self.metrics:
            metric.reset()

        first_batch_logged = False
        for imgs, masks in tqdm(loader, 
                        total=len(loader), 
                        desc=f"Evaluating {split}", 
                        leave=False,
                        colour='red'):
            imgs = imgs.to(self.device, non_blocking=True)
            masks = masks.long().to(self.device, non_blocking=True)
            
            with torch.autocast(device_type=self.device.type, dtype=torch.float16):
                _, logits = self.model(imgs)
                batch_loss = F.cross_entropy(logits, masks)

            cumulative_loss += batch_loss.item()
            pred = logits.argmax(dim=1)
            for metric in self.metrics:
                metric.update(pred, masks)

            if split == "val" and not first_batch_logged:
                self.log_segmentation_results(masks, pred)
                first_batch_logged = True
    
        mean_epoch_loss = cumulative_loss / len(loader)

        log_dict = {
            f'{split}/epoch_loss': mean_epoch_loss,
            'epoch': epoch
        }
        for metric in self.metrics:
            log_dict[f'{split}/{metric.__class__.__name__}'] = metric.compute()
    
        wandb.log(log_dict)

    def _save_model(self):
        save_path = self.path_ckpts / f"{self.config['wandb_run_name']}.pt"
        torch.save(self.model.state_dict(), save_path)
        print(f"Model saved to {save_path}")


if __name__ == '__main__':
    pass