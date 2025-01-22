import numpy as np
import random
import torch

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
