import argparse

import torch

from ml4cv_assignment.config.config import Config
from ml4cv_assignment.training.trainer import Trainer
from ml4cv_assignment.utils.misc import fix_random

torch.set_float32_matmul_precision("high")


def main(config_path: str) -> None:
    cfg = Config.load(config_path)
    fix_random(cfg.seed)

    trainer = Trainer(
        config=cfg.training,
        model=cfg.model,
        train_loader=cfg.train_dataloader,
        val_loader=cfg.val_dataloader,
    )
    trainer.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train segmentation model on Street Hazards"
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="./configs/base.yaml",
        help="Path to the YAML config file",
    )
    args = parser.parse_args()
    main(args.config_path)
