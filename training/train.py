from typing import Dict, Any
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
import wandb
from tqdm.notebook import tqdm

class Trainer:
    def __init__(self,
            config: Dict[str, Any],
            model: nn.Module,
            device: torch.device,
            train_loader: DataLoader,
            val_loader: DataLoader = None,
            metrics: list = []
        ) -> None:
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.model = model.to(device)

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

    def logfn(self, values: Dict[str, Any]) -> None:
        wandb.log(values, step=self.step, commit=False)

    def train(self, run_name) -> None:
        wandb.init(
            project=self.config['wandb']['project'],
            name=run_name,
            config=self.config
        )

        for epoch in tqdm(range(1, self.config['training']['num_epochs'] + 1), desc="Epoch", colour='green'):
            self.model.train()

            for batch_idx, (imgs, masks) in tqdm(enumerate(self.train_loader), 
                                             total=len(self.train_loader), 
                                             desc=f"Epoch {epoch} Batches", 
                                             leave=False,
                                             colour='blue'):
                loss = self._training_step(imgs, masks)

                # Logging

            # Validation step
            if self.val_loader:
                pass

        wandb.finish()

    def _training_step(self, imgs: torch.Tensor, masks: torch.Tensor) -> float:
        imgs = imgs.to(self.device)
        masks = masks.long().to(self.device) # Change masks type to long

        # Forward pass
        features, seg_map = self.model(imgs)
        loss = F.cross_entropy(seg_map, masks)

        # Logging
        # image_np = imgs[0].cpu().detach().numpy().transpose(1, 2, 0)  # From [C, H, W] to [H, W, C]
        # pred_mask = seg_map[0].cpu().detach().numpy().argmax(axis=0)  # From [num_classes, H, W] to [H, W]
        # gt_mask = masks[0].cpu().detach().numpy()  # already [H, W]

        # mask_img = wandb.Image(
        #     image_np,  # Transposed image
        #     masks={
        #         "predictions": {"mask_data": pred_mask, "class_labels": {0: "background", 1: "object"}},
        #         "ground_truth": {"mask_data": gt_mask, "class_labels": {0: "background", 1: "object"}},
        #     },
        # )
        # wandb.log({"mask_visualization": mask_img})

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