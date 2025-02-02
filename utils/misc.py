import numpy as np
import random
import torch
from torchinfo import summary


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


def print_summary(net, input_size, verbose=True):
    net_info = summary(net, input_size=input_size)
    params = net_info.total_params
    macs = net_info.total_mult_adds
    if verbose: print(net_info)
    print("\nNetwork's n°params: %.3fk \tMAC: %.3fM\n" % (params/1e3, macs/1e6))