import random
from datetime import datetime
from typing import Dict, Optional, Union

import numpy as np
import torch
from prettytable import PrettyTable


def resolve_device(device: Optional[Union[str, torch.device]] = None) -> torch.device:
    if isinstance(device, str):
        return torch.device(device)
    if isinstance(device, torch.device):
        return device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def fix_random(seed: int) -> None:
    """
    Fix all the possible sources of randomness.
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def generate_run_name(base_name: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{base_name}_{timestamp}"
    return run_name


def display_eval_results(results: Dict[str, float], title: str) -> None:
    table = PrettyTable()
    table.title = title
    table.field_names = ["Metric", "Value"]
    table.align["Metric"] = "l"
    table.align["Value"] = "r"

    for key, value in results.items():
        table.add_row([key, f"{value:.2f}"])

    print(table)
