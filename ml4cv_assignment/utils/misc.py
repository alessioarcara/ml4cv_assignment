import random
from datetime import datetime
from typing import Optional, Union

import numpy as np
import torch

# from tqdm.notebook import tqdm
# from ml4cv_assignment.models.detector import OpenSetSegmenter
# from ml4cv_assignment.models.old_decoder import ModelInfo

# from ml4cv_assignment.training.metrics import Metric


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


# def compute_metrics(
#    data_loader: torch.utils.data.DataLoader,
#    segmenter: OpenSetSegmenter,
#    metrics: List[Metric],
#    device="cuda" if torch.cuda.is_available() else "cpu",
# ):
#    for metric in metrics:
#        metric.reset()
#
#    for imgs, masks in tqdm(data_loader, desc="Computing metrics"):
#        imgs = imgs.to(device)
#        masks = masks.to(device)
#
#        closed_set_preds, open_set_probs = segmenter(imgs)
#
#        for metric in metrics:
#            metric.update(open_set_probs, closed_set_preds, masks)
#
#    results = {}
#    for metric in metrics:
#        results[metric.__class__.__name__] = metric.compute()
#        print(f"{metric.__class__.__name__}: {results[metric.__class__.__name__]}")
#
#    return results
#
