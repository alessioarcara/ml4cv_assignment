from typing import Dict, Any, Optional
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm

class Trainer:
    def __init__(self,
            config: Dict[str, Any],
            model: nn.Module,
            device: torch.device,
            train_loader: DataLoader,
            val_loader: DataLoader = None,
        ) -> None:
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.model = model.to(device)

        # Logging
        wandb.init(
            project=self.config.wandb_project,
            name=self.config.wandb_run_name,
            config=self.config
        )

        # Optimization
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )

        # Learning rate schedule
        num_steps = self.config['num_epochs'] * len(self.train_loader)
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.config.learning_rate,
            total_steps=num_steps,
            pct_start=0.1
        )

    def logfn(self, values: Dict[str, Any]) -> None:
        wandb.log(values, step=self.step, commit=False)

    def train(self) -> None:
        for epoch in tqdm(range(1, self.config['num_epochs'] + 1), desc="Epoch"):
            self.model.train()

            for batch_idx, (imgs, labels) in enumerate(self.train_loader):
                loss = self._training_step(imgs, labels)

                # Logging


            # Validation step
            if self.val_loader:
                pass

        wandb.finish()

    def _training_step(self, imgs: torch.Tensor, labels: torch.Tensor) -> float:
        imgs = imgs.to(self.device)
        labels = labels.to(self.device)

        # Forward pass
        pred = self.model(imgs)
        loss = F.cross_entropy(pred, labels)

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        return loss.item()

    @torch.no_grad()
    def eval(self, split: str) -> None:
        self.model.eval()
        loader = self.val_loader if split == "val" else self.train_loader

        with torch.no_grad():
            raise NotImplemented("Validation logic is not implemented yet.")

    def _save_model(self):
        path_ckpts = Path(self.config['save_dir'])
        path_ckpts.mkdir(exist_ok=True)
        save_path = path_ckpts / f"{self.config['wandb_run_name']}.pt"
        torch.save(self.model.state_dict(), save_path)
        print(f"Model saved to {save_path}")

if __name__ == '__main__':
    pass