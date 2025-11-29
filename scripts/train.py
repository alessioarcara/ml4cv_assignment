import argparse
import sys
from pathlib import Path
from typing import List

import torch
from loguru import logger

from ml4cv_assignment.config.config import Config
from ml4cv_assignment.config.utils import get_experiment_config
from ml4cv_assignment.training.trainer import Trainer
from ml4cv_assignment.utils.misc import fix_random

torch.set_float32_matmul_precision("high")


def main(config_paths: List[str]) -> None:
    paths = [Path(p) for p in config_paths]

    try:
        merged_config_dict = get_experiment_config(paths)
    except Exception as e:
        logger.error(f"❌ Error merging configs: {e}")
        sys.exit(1)

    try:
        cfg = Config.model_validate(merged_config_dict)
    except Exception as e:
        logger.error(f"❌ Configuration validation failed:\n{e}")
        sys.exit(1)

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
        description="Train segmentation model on Street Hazards"
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
