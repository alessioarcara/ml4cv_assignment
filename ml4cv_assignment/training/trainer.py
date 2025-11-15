from collections import defaultdict
from typing import Any, Dict, Mapping, Optional, Tuple, Union

import torch
import torch.nn as nn
from loguru import logger
from torch import Tensor, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from ml4cv_assignment.config.trainer_config import TrainerConfig
from ml4cv_assignment.training.metrics import MetricCollection
from ml4cv_assignment.utils.misc import resolve_device
from ml4cv_assignment.utils.typings import Stage, StepOutput


class Trainer:
    def __init__(
        self,
        config: TrainerConfig,
        model: nn.Module,
        train_loader: DataLoader,
        run_name: str,
        val_loader: Optional[DataLoader] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.callbacks = config.callbacks
        self.losses = config.losses
        self.metric_collection = MetricCollection(config.metrics)
        self.denormalize = config.denormalize
        self.run_name = run_name
        self.history: Dict[str, float] = {}
        self._stop_training = False

        # Model
        self.device = resolve_device(device)
        self.model = model.to(self.device)
        if self.config.use_torch_compile:
            self.model.compile()

        # Optimizer
        lr = self.config.lr
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=config.weight_decay
        )

        # Scheduler
        total_steps = self.config.num_epochs * len(self.train_loader)
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=lr * 3,
            total_steps=total_steps,
            pct_start=0.1,
            anneal_strategy="linear",
        )

        # AMP
        self.scaler = torch.amp.GradScaler(enabled=self.config.use_mixed_precision)

    @property
    def stop_training(self) -> bool:
        return self._stop_training

    @stop_training.setter
    def stop_training(self, value: bool) -> None:
        self._stop_training = value

    def _on_training_end(self) -> None:
        for callback in self.callbacks:
            callback.on_train_end(self)

    def _on_eval_end(self) -> None:
        for callback in self.callbacks:
            callback.on_eval_end(self)

    def get_loader(self, stage: Stage) -> Optional[DataLoader]:
        return self.train_loader if stage == Stage.TRAIN else self.val_loader

    def prepare_batch(self, batch: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        imgs, masks = batch
        imgs = imgs.to(self.device, non_blocking=True)
        masks = masks.long().to(self.device, non_blocking=True)
        return imgs, masks

    def _compute_loss(
        self, logits: Tensor, gt_masks: Tensor, stage: str
    ) -> Tuple[Tensor, Dict[str, float]]:
        losses = []
        loss_dict = {}

        for i, loss_fn in enumerate(self.losses):
            loss_i: Tensor = loss_fn(logits, gt_masks)
            losses.append(loss_i)
            loss_dict[f"{stage}/batch_loss_{i}"] = loss_i.item()

        total_loss = torch.mean(torch.stack(losses))
        loss_dict[f"{stage}/loss"] = total_loss.item()

        return total_loss, loss_dict

    def _prepare_input(
        self, data: Union[torch.Tensor, Any]
    ) -> Union[torch.Tensor, Any]:
        """
        Prepares one `data` before feeding it to the model, be it a tensor or a nested list/dictionary of tensors.
        """
        if isinstance(data, Mapping):
            return type(data)({k: self._prepare_input(v) for k, v in data.items()})  # type: ignore
        elif isinstance(data, (tuple, list)):
            return type(data)(self._prepare_input(v) for v in data)
        elif isinstance(data, torch.Tensor):
            return data.to(self.device, non_blocking=True)
        return data

    def _train_step(self, batch: Tuple[Tensor, Tensor]) -> StepOutput:
        # imgs, masks = self.prepare_batch(batch)
        inputs = self._prepare_input(batch)
        # inputs = batch

        self.optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16,
            enabled=self.config.use_mixed_precision,
        ):
            outputs = self.model(inputs, return_preds=False)

            # logits = self.model(imgs)
            # total_loss, loss_dict = self._compute_loss(logits, masks, Stage.TRAIN)
        loss = outputs["loss"]

        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()

        # return loss_dict
        return {"train/batch_loss": loss.item()}

    def _eval_step(self, batch: Tuple[Tensor, Tensor], stage: Stage) -> StepOutput:
        inputs = self._prepare_input(batch)
        # imgs, masks = self.prepare_batch(batch)

        with torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16,
            enabled=self.config.use_mixed_precision,
        ):
            # logits: Tensor = self.model(imgs)
            # _, loss_dict = self._compute_loss(logits, masks, stage)
            outputs = self.model(inputs)

        pred_masks = outputs["preds"]
        gt_masks = inputs["orig_masks"]
        # pred = logits.argmax(dim=1)
        self.metric_collection.update(None, pred_masks, gt_masks)

        # return loss_dict
        return {f"{stage}/loss": outputs["loss"].item()}

    def run(self) -> None:
        wandb.init(
            project=self.config.wandb_project_name,
            entity=self.config.wandb_entity,
            name=self.run_name,
            config=self.config.model_dump(),
        )
        wandb.watch(self.model, log="all", log_freq=100)

        try:
            for epoch in tqdm(
                range(1, self.config.num_epochs + 1), desc="Epoch", colour="green"
            ):
                self.model.train()
                train_loader = self.get_loader(Stage.TRAIN)
                assert train_loader is not None

                for batch in tqdm(
                    train_loader,
                    total=len(train_loader),
                    desc=f"Epoch {epoch} Batches",
                    leave=False,
                    colour="blue",
                ):
                    batch_log = self._train_step(batch)
                    batch_log["train/lr"] = self.scheduler.get_last_lr()[0]
                    wandb.log(batch_log)

                if epoch % self.config.evaluation_rate == 0:
                    train_results = self.eval(Stage.TRAIN)
                    val_results = self.eval(Stage.VAL)

                    epoch_log = {}
                    epoch_log.update(train_results)
                    epoch_log.update(val_results)

                    self.history.update(epoch_log)

                    wandb.log(epoch_log)

                    self._on_eval_end()

                    if self.stop_training:
                        logger.info("Early stopping triggered! No improvement.")
                        break

            self._on_training_end()

        finally:
            wandb.finish()

    @torch.inference_mode()
    def eval(self, stage: Stage) -> Dict[str, float]:
        self.model.eval()
        loader = self.get_loader(stage)

        if loader is None:
            logger.warning(
                "No data to evaluate for the '{}' stage. Skipping evaluation.", stage
            )
            return {}

        self.metric_collection.reset()
        loss_acc: defaultdict[str, float] = defaultdict(float)
        num_batches = len(loader)

        for batch in tqdm(
            loader,
            total=num_batches,
            desc=f"Evaluating {stage}",
            leave=False,
            colour="red",
        ):
            loss_dict = self._eval_step(batch, stage)
            for k, v in loss_dict.items():
                loss_acc[f"{k}_epoch"] += v

        losses_avg: Dict[str, float] = {}
        for k, v in loss_acc.items():
            losses_avg[k] = v / num_batches

        metrics_dict = self.metric_collection.compute()

        eval_results = {
            **losses_avg,
            **{f"{stage}/{k}_epoch": v for k, v in metrics_dict.items()},
        }

        return eval_results
