import argparse
from typing import Any, Dict, List, Literal, Optional

import torch

from ml4cv_assignment.config.utils import build_config
from ml4cv_assignment.evaluation import evaluate_split
from ml4cv_assignment.utils.misc import print_eval_results


def main(
    config_paths: List[str],
    split: Literal["val", "test"],
    checkpoint_path: Optional[str],
) -> None:
    torch.set_float32_matmul_precision("high")

    override_config: Dict[str, Any] = {}

    # If the user wants to evaluate a specific checkpoint, override the config
    if checkpoint_path:
        override_config["paths"] = {"checkpoint": checkpoint_path}

    # Enforces the injection of anomalies only for only the validation set to
    # serve as a proxy for OoD performance evaluation
    if split == "val":
        override_config["dataset_config"] = {"add_anomalies": True, "prob_insert": 1.0}
    elif split == "test":
        override_config["dataset_config"] = {"add_anomalies": False}
    else:
        raise ValueError(f"Invalid split: {split}. Must be 'val' or 'test'.")

    cfg, _ = build_config(config_paths, override_config)

    results = evaluate_split(cfg=cfg, split=split)
    print_eval_results(results, title=f"{split.capitalize()} Set Evaluation")


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
    parser.add_argument(
        "--split",
        type=str,
        choices=["val", "test"],
        default="test",
        help="Dataset split to evaluate on (default: test)",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Path to a specific model checkpoint to load",
    )
    args = parser.parse_args()

    main(args.configs, args.split, args.checkpoint_path)
