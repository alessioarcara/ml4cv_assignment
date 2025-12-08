from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor


class BaseModel(nn.Module, ABC):
    """
    Base class for all models.

    Subclasses must implement:
    1. `get_param_groups`: To return parameter groups for the optimizer.
    2. `_forward_impl`: To define the specific inference logic.

    Subclasses may optionally override:
    - `on_epoch_start`
    - `on_train_epoch_start`
    - `on_train_epoch_end`
    """

    def __init__(self, losses: Optional[List[nn.Module]] = None) -> None:
        super().__init__()
        self.losses = nn.ModuleList(losses) if losses else nn.ModuleList()

    @abstractmethod
    def get_param_groups(self) -> List[Dict[str, Any]]: ...

    @abstractmethod
    def _forward_impl(
        self, inputs: Dict[str, Tensor], return_preds: bool
    ) -> Dict[str, Any]: ...

    def on_train_epoch_start(self) -> None:
        """
        Hook called at training epoch start.
        """
        pass

    def on_train_epoch_end(self) -> None:
        """
        Hook called at training epoch end.
        """
        for loss_fn in self.losses:
            hook = getattr(loss_fn, "on_epoch_end", None)

            if hook is None and hasattr(loss_fn, "loss"):
                hook = getattr(loss_fn.loss, "on_epoch_end", None)

            if callable(hook):
                hook()

    def on_eval_start(self) -> None:
        """
        Hook called at evaluation start.
        """
        pass

    def _compute_loss(
        self, logits: Tensor, gt_masks: Tensor
    ) -> Tuple[Tensor, Dict[str, float]]:
        loss_vals = []
        loss_dict = {}

        for i, loss_fn in enumerate(self.losses):
            loss_i: Tensor = loss_fn(logits, gt_masks)
            loss_vals.append(loss_i)

            name = getattr(loss_fn, "__name__", f"loss_{i}")
            loss_dict[f"batch_{name}"] = loss_i.item()

        if not loss_vals:
            return torch.tensor(0.0, device=logits.device, requires_grad=True), {}

        total_loss = torch.mean(torch.stack(loss_vals))
        return total_loss, loss_dict

    def forward(
        self, inputs: Dict[str, Tensor], return_preds: bool = True
    ) -> Dict[str, Any]:
        """
        Template method for forward pass.

        1. Delegates inference to the subclass's `_forward_impl`.
        2. Automatically calculate loss if conditions are met.
        """
        outputs = self._forward_impl(inputs, return_preds=return_preds)

        # Compute loss only if:
        # - There are loss functions defined
        # - The subclass produced 'logits' in outputs
        if self.losses and "logits" in outputs:
            targets = inputs.get("orig_masks")

            if targets is not None:
                total_loss, loss_dict = self._compute_loss(outputs["logits"], targets)

                outputs.update(loss_dict)

                if "loss" in outputs:
                    outputs["loss"] += total_loss
                else:
                    outputs["loss"] = total_loss

        return outputs

    def freeze_module(self, module: nn.Module) -> None:
        """
        Freezes the parameters of a specified module.
        """
        for param in module.parameters():
            param.requires_grad = False
        module.eval()

    def unfreeze_module(self, module: nn.Module) -> None:
        """
        Unfreezes the parameters of a specified module.
        """
        for param in module.parameters():
            param.requires_grad = True
        module.train()
