import argparse
import sys
from typing import List

import torch
from loguru import logger

from ml4cv_assignment.config.utils import build_config
from ml4cv_assignment.training.trainer import Trainer
from ml4cv_assignment.utils.misc import fix_random

torch.set_float32_matmul_precision("high")


def main(config_paths: List[str]) -> None:
    cfg, merged_config_dict = build_config(config_paths)

    fix_random(cfg.seed)

    trainer = Trainer(
        config=cfg.training,
        model=cfg.model,
        train_loader=cfg.train_dataloader,
        val_loader=cfg.val_dataloader,
        experiment_raw_config=merged_config_dict,
    )

    try:
        logger.info("🚀 Starting Training ...")
        trainer.train()
    except KeyboardInterrupt:
        logger.warning("⚠️ Training interrupted by user.")
        sys.exit(0)
    except Exception as e:
        logger.exception(f"❌ Unhandled exception during training: {e}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train segmentation model on StreetHazards"
    )
    parser.add_argument(
        "--configs",
        type=str,
        nargs="+",
        required=True,
        help="Paths to the YAML config files",
    )
    args = parser.parse_args()
    main(args.configs)
