import argparse
from typing import List

import torch

from ml4cv_assignment.config.utils import build_config
from ml4cv_assignment.training.trainer import Trainer
from ml4cv_assignment.utils.misc import display_eval_results, fix_random
from ml4cv_assignment.utils.typings import Stage

torch.set_float32_matmul_precision("high")


def main(config_paths: List[str]) -> None:
    cfg, merged_config_dict = build_config(config_paths)

    fix_random(cfg.seed)

    trainer = Trainer(
        config=cfg.training,
        model=cfg.model,
        test_loader=cfg.test_dataloader,
        experiment_raw_config=merged_config_dict,
    )

    display_eval_results(trainer.eval(Stage.TEST), title="Test Set Evaluation")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluation segmentation model on StreetHazards"
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
