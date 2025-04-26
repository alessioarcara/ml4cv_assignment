import random
from datetime import datetime
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import yaml
from torchinfo import summary

from models.model import ModelInfo


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r") as file:
        cfg = yaml.safe_load(file)
    return cfg


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("🚀 CUDA device is available!")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("🍎 MPS device is available!")
    else:
        device = torch.device("cpu")
        print("🐢 No GPU available. Falling back to the CPU.")
    return device


def fix_random(seed: int) -> None:
    """Fix all the possible sources of randomness.

    Args:
        seed: the seed to use.
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def generate_run_name(
    config: Dict[str, Any],
    model_info: ModelInfo,
    criterions: List[Tuple[float, nn.Module]],
) -> str:
    encoder_name = config["model"]["encoder_name"]
    d = config["model"]["d"]
    imgH = config["training"]["img_height"]
    imgW = config["training"]["img_width"]

    loss_str = ",".join([l.__class__.__name__ for _, l in criterions])
    atrous_str = "-".join(map(str, model_info.atrous_rates))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = (
        f"ENC_{encoder_name}"
        f"_DEC_{len(model_info.fpn_features) - 1}FAM{d}"
        f"_ASPP{atrous_str}"
        f"_IMG{imgH}x{imgW}"
        f"_LOSS_{loss_str}"
        f"_{timestamp}"
    )
    return run_name


def print_summary(net, input_size, verbose=True):
    net_info = summary(net, input_size=input_size)
    params = net_info.total_params
    macs = net_info.total_mult_adds
    if verbose:
        print(net_info)
    print("\nNetwork's n°params: %.3fk \tMAC: %.3fM\n" % (params / 1e3, macs / 1e6))
