from pathlib import Path
from typing import Any, Callable, Dict, Optional

import torch
import torch.nn as nn
from loguru import logger
from torch import optim
from torch.profiler import ProfilerActivity, profile, record_function, schedule
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb

from .metrics import Metric
from .monitor import WandbMonitor


class Trainer:
    def __init__(
        self,
        config: Dict[str, Any],
        model: nn.Module,
        device: torch.device,
        train_loader: DataLoader,
        criterions: list[tuple[float, Callable]],
        denorm: Callable,
        val_loader: Optional[DataLoader] = None,
        class_dict: Dict[int, str] = {},
        metrics: list[Metric] = [],
        use_amp: bool = True,
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

        # Monitor
        self.monitor = WandbMonitor(
            config=config, metrics=metrics, denorm=denorm, class_dict=class_dict
        )

        # Optimzier
        lr = self.config["training"]["lr"]
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=self.config["training"]["wd"]
        )

        # Scheduler
        num_steps = self.config["training"]["num_epochs"] * len(self.train_loader)
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=lr,
            total_steps=num_steps,
            pct_start=0.1,
        )

        # AMP
        self.scaler = torch.amp.GradScaler() if use_amp else None

        # Training state
        self.step = 0
        self.epoch = 0
        self.best_miou = 0.0

        # Save dir
        save_dir = Path.home() / config["paths"]["save_path"]
        save_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir = save_dir
        self.evaluation_rate = config["training"].get("evaluation_rate", 1)

    def train(self, run_name: str) -> None:
        with self.monitor.run_context(run_name, len(self.criterions)):
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                schedule=schedule(wait=1, warmup=1, active=1, repeat=1),
                on_trace_ready=torch.profiler.tensorboard_trace_handler("saved/logs"),
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
            ) as p:
                for epoch in tqdm(
                    range(1, self.config["training"]["num_epochs"] + 1),
                    desc="Epoch",
                    colour="green",
                ):
                    self.model.train()
                    for batch in tqdm(
                        self.train_loader,
                        total=len(self.train_loader),
                        desc=f"Epoch {epoch} Batches",
                        leave=False,
                        colour="blue",
                    ):
                        self._training_step(*batch, profiler=p)

                    if epoch % self.evaluation_rate == 0:
                        self.eval("train", epoch)
                        if self.val_loader:
                            self.eval("val", epoch)

    def _training_step(
        self, imgs: torch.Tensor, masks: torch.Tensor, profiler=None
    ) -> None:
        imgs = imgs.to(self.device, non_blocking=True)
        masks = masks.long().to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)

        if self.use_amp:
            with torch.autocast(device_type=self.device.type, dtype=torch.float16):
                logits = self.model(imgs)
                batch_loss = self._compute_loss(logits, masks, True)

            self.scaler.scale(batch_loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            logits = self.model(imgs)
            batch_loss = self._compute_loss(logits, masks, True)
            batch_loss.backward()
            self.optimizer.step()

        self.scheduler.step()
        self.step += 1

        if self.step % 10 == 0:
            self.monitor.log_training_step(
                batch_loss.item(), self.scheduler.get_last_lr()[0], self.step
            )

        if profiler is not None:
            profiler.step()

    def _compute_loss(
        self,
        logits: torch.Tensor,
        masks: torch.Tensor,
        is_train=False,
    ) -> torch.Tensor:
        """
        Compute a weighted sum of all provided criterions.
        """
        with record_function("loss"):
            total_loss = 0.0
            batch_losses = {}

            for i, (weight, loss_fn) in enumerate(self.criterions):
                loss = loss_fn(logits, masks)
                total_loss += weight * loss
                batch_losses[f"train/batch_loss{i}"] = loss.item()

            if is_train:
                self.monitor.log_batch_losses(batch_losses, self.step)

            return total_loss

    @torch.inference_mode()
    def eval(self, split: str, epoch: int) -> None:
        """
        Evaluate the model on either the training loader or validation loader.
        """
        self.model.eval()
        loader = self.val_loader if split == "val" else self.train_loader

        total_loss = 0.0
        for metric in self.metrics:
            metric.reset()

        first_batch_logged = False
        for imgs, masks in tqdm(
            loader,
            total=len(loader),
            desc=f"Evaluating {split}",
            leave=False,
            colour="red",
        ):
            imgs = imgs.to(self.device, non_blocking=True)
            masks = masks.long().to(self.device, non_blocking=True)

            if self.use_amp:
                with torch.autocast(device_type=self.device.type, dtype=torch.float16):
                    logits = self.model(imgs)
                    batch_loss = self._compute_loss(logits, masks)
            else:
                logits = self.model(imgs)
                batch_loss = self._compute_loss(logits, masks)

            total_loss += batch_loss.item()
            pred = logits.argmax(dim=1)

            for metric in self.metrics:
                metric.update(logits, pred, masks)

            if split == "val" and not first_batch_logged:
                self.monitor.log_segmentation_results(imgs, masks, pred)
                self.monitor.log_pixel_embeddings(logits, pred, masks)
                first_batch_logged = True

        mean_loss = total_loss / len(loader)

        log_dict = {f"{split}/epoch_totalloss": mean_loss, "epoch": epoch}

        for metric in self.metrics:
            metric_name = metric.__class__.__name__
            log_dict[f"{split}/{metric_name}"] = metric.compute()

        self.monitor.log_metrics(log_dict)

        if split == "val":
            curr_miou = None
            for key, value in log_dict.items():
                if "meaniou" in key.lower():
                    curr_miou = value
                    break
            if curr_miou is not None and curr_miou > self.best_miou:
                self.best_miou = curr_miou
                self._save_model(epoch, curr_miou)

    def _save_model(self, epoch: int, miou: float) -> None:
        """
        Save model checkpoint to the configured directory.
        """
        filename = f"{wandb.run.name}_epoch{epoch}_miou{miou:.4f}.pt"
        path = self.ckpt_dir / filename
        torch.save(self.model.state_dict(), path)
        logger.info(f"Saved checkpoint to {path}")
