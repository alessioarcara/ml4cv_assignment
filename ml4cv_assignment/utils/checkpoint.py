from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from loguru import logger

from ml4cv_assignment.utils.typings import StateDict


def _remap_state_dict_keys(
    state_dict: StateDict, remap_rules: Dict[str, str]
) -> StateDict:
    """
    Remaps keys in a state_dict according to replacement rules.
    """
    if not remap_rules:
        return state_dict

    logger.info(f"Remapping checkpoint keys with rules: {remap_rules}")
    keys = list(state_dict.keys())

    for k in keys:
        new_key = k

        for old, new in remap_rules.items():
            if old in new_key:
                new_key = new_key.replace(old, new)

        if new_key != k:
            state_dict[new_key] = state_dict.pop(k)
            logger.debug(f"Renamed: {k} -> {new_key}")

    return state_dict


def _filter_state_dict(model: nn.Module, state_dict: StateDict) -> StateDict:
    model_state = model.state_dict()
    filtered_dict = {}

    for k, v in state_dict.items():
        if k in model_state and v.shape != model_state[k].shape:
            logger.warning(
                f"⚠️ Shape mismatch for {k}: ckpt {v.shape} vs model {model_state[k].shape}. Skipping."
            )
            continue
        filtered_dict[k] = v

    return filtered_dict


def load_checkpoint(model: nn.Module, path: Path, remap_rules: dict[str, str]) -> None:
    ckpt = torch.load(path, map_location="cpu")

    state_dict = _remap_state_dict_keys(ckpt, remap_rules)

    safe_state_dict = _filter_state_dict(model, state_dict)

    missing_keys, unexpected_keys = model.load_state_dict(safe_state_dict, strict=False)

    if len(missing_keys) > 0:
        logger.warning(f"Missing keys: {missing_keys}")
    if len(unexpected_keys) > 0:
        logger.warning(f"Unexpected keys: {unexpected_keys}")

    logger.info("✅ Checkpoint loaded successfully")
