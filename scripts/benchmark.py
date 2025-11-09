import time

import numpy as np
import torch
from loguru import logger

from ml4cv_assignment.models.model import build_model
from ml4cv_assignment.utils.misc import get_device, load_config


class Timer:
    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.start = time.time()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.interval = time.time() - self.start


def benchmark(model, input_tensor, n_iter=30, warmup_iter=20):
    logger.info("Running warmup...")
    with torch.no_grad():
        for _ in range(warmup_iter):
            with Timer():
                _ = model(input_tensor)

    logger.info("Running benchmark...")
    times = []
    with torch.no_grad():
        for _ in range(n_iter):
            with Timer() as timer:
                _ = model(input_tensor)
            times.append(timer.interval)

    mean_time = float(np.mean(times))
    std_time = float(np.std(times))
    logger.info("Benchmark results -- mean: {} s, std: {} s", mean_time, std_time)
    return mean_time, std_time


def main():
    config = load_config("./config.yaml")
    imgH = config["training"]["img_height"]
    imgW = config["training"]["img_width"]
    bs = config["training"]["batch_size"]
    n_classes = 13

    device = get_device()
    input_tensor = torch.rand((bs, 3, imgH, imgW)).to(device)
    logger.debug(f"Input tensor shape: {input_tensor.shape}")

    model, _ = build_model(config, n_classes)
    model.eval()
    model.to(device)
    _ = benchmark(model, input_tensor)

    optimized_model = torch.compile(
        model.encoder, mode="reduce-overhead", fullgraph=True
    )
    optimized_model.eval()
    optimized_model.to(device)
    _ = benchmark(optimized_model, input_tensor)


if __name__ == "__main__":
    main()
