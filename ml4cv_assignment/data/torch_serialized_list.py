import pickle
from typing import Any, List

import numpy as np
import torch
from loguru import logger


# Implementation based on Detectron2:
# https://github.com/facebookresearch/detectron2/blob/main/detectron2/data/common.py
class TorchSerializedList:
    """
    A list-like object whose items are serialized and stored in a torch tensor. When
    launching a process that uses TorchSerializedList with "fork" start method,
    the subprocess can read the same buffer without triggering copy-on-access. When
    launching a process that uses TorchSerializedList with "spawn/forkserver" start
    method, the list will be pickled by a special ForkingPickler registered by PyTorch
    that moves data to shared memory. In both cases, this allows parent and child
    processes to share RAM for the list data, hence avoids the issue in
    https://github.com/pytorch/pytorch/issues/13246.

    See also https://ppwwyyxx.com/blog/2022/Demystify-RAM-Usage-in-Multiprocess-DataLoader/
    on how it works.
    """

    def __init__(self, lst: list):
        """
        Serialize a list of Python objects into a single contiguous torch tensor
        and store cumulative addresses for indexing.
        """
        self._lst = lst

        def _serialize(data: Any) -> np.ndarray:
            buffer = pickle.dumps(data, protocol=-1)
            return np.frombuffer(buffer, dtype=np.uint8)

        serialized_list: List[np.ndarray] = [_serialize(x) for x in lst]

        lengths = np.array([len(x) for x in serialized_list], dtype=np.int64)
        self._addr = torch.from_numpy(np.cumsum(lengths))

        self._serialized = torch.from_numpy(np.concatenate(serialized_list))
        logger.info(
            "Serialized dataset takes {:.2f} MiB".format(len(self._lst) / 1024**2)
        )

    def __len__(self):
        return len(self._addr)

    def __getitem__(self, idx: int) -> Any:
        start_addr: int = 0 if idx == 0 else int(self._addr[idx - 1].item())
        end_addr: int = int(self._addr[idx].item())

        bytes_view = memoryview(self._serialized[start_addr:end_addr].numpy().tobytes())
        return pickle.loads(bytes_view)
