from collections import defaultdict
from typing import Any, Dict, Mapping, Optional, Union

import torch
from loguru import logger
from torch import Tensor, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from ml4cv_assignment.config.trainer_config import TrainerConfig
from ml4cv_assignment.models.model import BaseModel
from ml4cv_assignment.training.metrics import MetricCollection
from ml4cv_assignment.utils.misc import generate_run_name, resolve_device
from ml4cv_assignment.utils.typings import Batch, Stage, StepOutput


class Trainer:
    def __init__(
        self,
        config: TrainerConfig,
        model: BaseModel,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.callbacks = config.callbacks
        self.metric_collection = MetricCollection(config.metrics)
        self.denormalize = config.denormalize
        self.history: Dict[str, float] = {}
        self._stop_training = False

        # Model
        self.device = resolve_device(device)
        self.model = model.to(self.device)
        if self.config.use_torch_compile:
            self.model.compile()

        # Optimizer
        param_groups = self.model.get_param_groups()
        self.optimizer = optim.AdamW(param_groups, fused=True)

        # Scheduler
        max_lrs = [float(group["lr"]) * 3 for group in param_groups]
        total_steps = self.config.num_epochs * len(self.train_loader)
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=max_lrs,
            total_steps=total_steps,
            pct_start=0.3,
            anneal_strategy="cos",
        )

        # AMP
        self.scaler = torch.amp.GradScaler(enabled=self.config.use_mixed_precision)

    @property
    def stop_training(self) -> bool:
        return self._stop_training

    @stop_training.setter
    def stop_training(self, value: bool) -> None:
        self._stop_training = value

    def _run_callbacks(self, hook_name: str) -> None:
        """
        A helper function to run a specific hook on all callbacks
        """
        for callback in self.callbacks:
            method_to_call = getattr(callback, hook_name, None)
            if method_to_call:
                method_to_call(self)

    def _on_training_start(self) -> None:
        self._run_callbacks("on_train_start")

    def _on_training_end(self) -> None:
        self._run_callbacks("on_train_end")

    def _on_eval_end(self) -> None:
        self._run_callbacks("on_eval_end")

    def get_loader(self, stage: Stage) -> Optional[DataLoader]:
        return self.train_loader if stage == Stage.TRAIN else self.val_loader

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

    def _collect_losses(self, outputs: Dict[str, Any], prefix: str) -> Dict[str, float]:
        loss_dict: Dict[str, float] = {}

        for k, v in outputs.items():
            if "loss" in k:
                loss_dict[f"{prefix}/{k}"] = (
                    v.item() if isinstance(v, torch.Tensor) else float(v)
                )

        return loss_dict

    def _log_scheduler_lrs(self, log_dict: Dict[str, float]) -> None:
        lrs = self.scheduler.get_last_lr()
        for i, lr in enumerate(lrs):
            log_dict[f"train/lr_group_{i}"] = lr

    def _train_step(self, batch: Batch) -> StepOutput:
        inputs = self._prepare_input(batch)

        self.optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16,
            enabled=self.config.use_mixed_precision,
        ):
            outputs = self.model(inputs, return_preds=False)

        loss = outputs["loss"]

        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()

        return self._collect_losses(outputs, Stage.TRAIN)

    def _eval_step(self, batch: Batch, stage: Stage) -> StepOutput:
        inputs: Dict[str, Tensor] = self._prepare_input(batch)  # type: ignore

        with torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16,
            enabled=self.config.use_mixed_precision,
        ):
            outputs = self.model(inputs)

        pred_masks = outputs["preds"]
        gt_masks = inputs["orig_masks"]

        self.metric_collection.update(None, pred_masks, gt_masks)  # type: ignore

        return self._collect_losses(outputs, stage)

    def run(self) -> None:
        wandb.init(
            project=self.config.wandb_project_name,
            entity=self.config.wandb_entity,
            name=generate_run_name(self.config.wandb_base_run_name),
            config=self.config.model_dump(),
        )
        wandb.watch(self.model, log="all", log_freq=100)

        self._on_training_start()

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
                    self._log_scheduler_lrs(batch_log)
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
