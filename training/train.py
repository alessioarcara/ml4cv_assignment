from typing import Dict, Any, Callable
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
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from pathlib import Path


class Trainer:
    def __init__(self,
            config: Dict[str, Any],
            model: nn.Module,
            device: torch.device,
            train_loader: DataLoader,
            criterions: list[tuple[float, Callable]],
            val_loader: DataLoader = None,
            class_dict: Dict[int, str] = {},
            metrics: list[Metric] = [],
            use_amp: bool = True
        ) -> None:
        self.config = config
        self.criterions = criterions
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.model = model.to(device)
        self.class_dict = class_dict 
        self.metrics = metrics
        self.use_amp = use_amp

        # Optimization
        lr = self.config['training']['lr']
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=self.config['training']['wd']
        )

        # Scheduler
        num_steps = self.config['training']['num_epochs'] * len(self.train_loader)
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=lr,
            total_steps=num_steps,
            pct_start=0.1,
        )

        # AMP
        self.scaler = torch.amp.GradScaler() if use_amp else None

        self.step = 0
        self.epoch = 0
        self.best_miou = 0.0

        self.path_ckpts = Path.home() / self.config['paths']['save_path']
        self.path_ckpts.mkdir(parents=True, exist_ok=True)
        self.evaluation_rate = self.config['training'].get('evaluation_rate', 1)

    def log_segmentation_results(
        self,
        true: torch.Tensor,
        pred: torch.Tensor,
    ) -> None:
        """
        Log a side-by-side comparison of true vs predicted masks.
        """
        table = wandb.Table(columns=["Comparison"])
        
        for true_mask, pred_mask in zip(true, pred):
            true_np = true_mask.cpu().numpy()
            pred_np = pred_mask.cpu().numpy()
            
            true_colored = color(true_np, COLORS)
            pred_colored = color(pred_np, COLORS)
            
            comparison = np.concatenate((true_colored, pred_colored), axis=1)
            table.add_data(wandb.Image(comparison))
            
        wandb.log({"Segmentation Results": table})

    def log_pixel_embeddings(
            self,
            logits: torch.Tensor,
            pred: torch.Tensor,
            true: torch.Tensor,
            min_samples: int = 1000
        ) -> None:
            """
            Log PCA visualization of pixels embeddings for the first image in a batch.
            """
            C = logits.shape[1]
            embeddings = logits[0].permute(1, 2, 0).reshape(-1, C).cpu().numpy()
            labels = true[0].cpu().numpy().flatten()
            pred = pred[0].cpu().numpy().flatten()

            indices = []
            for cls in np.unique(labels):
                cls_indices = np.where(labels == cls)[0]
                if len(cls_indices) > 0:
                    n_samples = min(len(cls_indices), min_samples)
                    indices.extend(np.random.choice(cls_indices, size=n_samples, replace=False))
            
            anchors = np.eye(C) * self.config['training']['anchors_magnitude']
    
            selected_embeddings = embeddings[indices]
            selected_classes = labels[indices]
            combined_embeddings = np.vstack([selected_embeddings, anchors])

            pca = PCA(n_components=2, random_state=self.config['seed'])
            points_2d = pca.fit_transform(combined_embeddings)
            data_points = points_2d[:-C]
            anchor_points = points_2d[-C:]

            plt.figure(figsize=(12, 8))
            for cls in np.unique(selected_classes):
                mask = selected_classes == cls
                plt.scatter(data_points[mask, 0], data_points[mask, 1],
                        label=self.class_dict[cls], alpha=0.6, s=10)
                
            plt.scatter(anchor_points[:, 0], anchor_points[:, 1],
                        marker='x', s=300, c='red', label='Anchors')
            
            # Annotate anchors
            for cls in range(C):
                plt.annotate(f'{self.class_dict.get(cls)}',
                            (anchor_points[cls, 0], anchor_points[cls, 1]),
                            xytext=(0, 5), textcoords='offset points',
                            ha='center', fontsize=12, color='black', weight='bold')
            
            plt.title('PCA visualization of pixel embeddings with anchors', fontsize=16)
            plt.axis('off')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
            plt.tight_layout()
    
            wandb.log({"pixel_embeddings_pca": wandb.Image(plt)})
            plt.close()

    def _define_metrics(self) -> None:
        """
        Define all metrics to be tracked in Weights & Biases.
        """
        # Step-level metrics
        wandb.define_metric("train/lr", step_metric="step")
        wandb.define_metric("train/batch_totalloss", step_metric="step")
        for i in range(len(self.criterions)):
            wandb.define_metric(f"train/batch_loss{i}", step_metric="step")
        
        # Epoch-level metrics
        wandb.define_metric("train/epoch_totalloss", step_metric="epoch")
        wandb.define_metric("val/epoch_totalloss", step_metric="epoch")
        for metric in self.metrics:
            metric_name = metric.__class__.__name__
            wandb.define_metric(f"train/{metric_name}", step_metric="epoch")
            wandb.define_metric(f"val/{metric_name}", step_metric="epoch")

    def train(self, run_name) -> None:
        wandb.init(
            project=self.config['wandb']['project'],
            name=run_name,
            config=self.config
        )
        self.run_name = run_name
        self._define_metrics()

        try:
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
        finally:
            wandb.finish()

    def _compute_loss(
            self, 
            logits: torch.Tensor, 
            masks: torch.Tensor, 
            prelogits: torch.Tensor, 
            is_train = False, 
        ) -> torch.Tensor:
        """
        Compute a weighted sum of all provided criterions.
        """
        total_loss = 0.0
        batch_losses = {}

        for i, (weight, loss_fn) in enumerate(self.criterions):
            l = loss_fn(logits, masks)
            total_loss += weight * l

            batch_losses[f'train/batch_loss{i}'] = l.item()

        if is_train:
            wandb.log({
                **batch_losses,
                'step': self.step
            })

        return total_loss

    def _training_step(self, imgs: torch.Tensor, masks: torch.Tensor) -> float:
        imgs = imgs.to(self.device, non_blocking=True)
        masks = masks.long().to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)

        if self.use_amp:
            with torch.autocast(device_type=self.device.type, dtype=torch.float16):
                prelogits, logits = self.model(imgs)
                batch_loss = self._compute_loss(logits, masks, prelogits, True)

            self.scaler.scale(batch_loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            prelogits, logits = self.model(imgs)
            batch_loss = self._compute_loss(logits, masks, prelogits, True)
            batch_loss.backward()
            self.optimizer.step()

        self.scheduler.step()
        self.step += 1 
        
        if self.step % 10 == 0:
            wandb.log({
                'train/batch_totalloss': batch_loss.item(),
                'train/lr': self.scheduler.get_last_lr()[0],
                'step': self.step
            })

    @torch.inference_mode()
    def eval(self, split: str, epoch: int) -> None:
        """
        Evaluate the model on either the training loader or validation loader.
        """
        self.model.eval()
        loader = self.val_loader if split == "val" else self.train_loader

        cumulative_totalloss = 0.0
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
            
            if self.use_amp:
                with torch.autocast(device_type=self.device.type, dtype=torch.float16):
                    prelogits, logits = self.model(imgs)
                    batch_loss = self._compute_loss(logits, masks, prelogits)
            else:
                prelogits, logits = self.model(imgs)
                batch_loss = self._compute_loss(logits, masks, prelogits)

            cumulative_totalloss += batch_loss.item()
            pred = logits.argmax(dim=1)
            for metric in self.metrics:
                metric.update(pred, masks)

            if split == "val" and not first_batch_logged:
                self.log_segmentation_results(masks, pred)
                self.log_pixel_embeddings(logits, pred, masks)
                first_batch_logged = True
    
        mean_epoch_loss = cumulative_totalloss / len(loader)

        log_dict = {
            f'{split}/epoch_totalloss': mean_epoch_loss,
            'epoch': epoch
        }

        for metric in self.metrics:
            metric_name = metric.__class__.__name__
            log_dict[f'{split}/{metric_name}'] = metric.compute()
    
        wandb.log(log_dict)

        if split == "val":
            curr_miou = None
            for key, value in log_dict.items():
                if "meaniou" in key.lower():
                    curr_miou = value
                    break
            if curr_miou is not None and curr_miou > self.best_miou:
                self.best_miou = curr_miou 
                self._save_model(epoch, curr_miou)

    def _save_model(self, epoch: int, current_miou: float) -> None:
        """
        Save model checkpoint to the configured directory.
        """
        save_path = self.path_ckpts / f"{self.run_name}_epoch{epoch}_miou{current_miou:.4f}.pt"
        torch.save(self.model.state_dict(), save_path)
        print(f"Model saved to {save_path}")

if __name__ == '__main__':
    pass
