import argparse
import os
from pathlib import Path

import cv2
import torch
from kornia.losses import FocalLoss
from loguru import logger

# ── Project modules ───────────────────────────────────────────────────────────
from ml4cv_assignment.data.multi_epochs_dataloader import MultiEpochsDataLoader
from ml4cv_assignment.data.streethazards import (
    STREET_HAZARDS_CLASSES,
    StreetHazards,
    get_classes_as_dict,
)
from ml4cv_assignment.data.transforms import Denormalize, get_data_transforms
from ml4cv_assignment.models.model import build_model
from ml4cv_assignment.training.losses import (
    ObjectosphereLoss,
    PrototypicalGlobalLocalTripletLoss,
)
from ml4cv_assignment.training.metrics import MeanIoU
from ml4cv_assignment.training.trainer import Trainer
from ml4cv_assignment.utils.misc import (
    fix_random,
    generate_run_name,
    get_device,
    load_config,
    print_summary,
)

# ── Argument parsing ──────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="Train segmentation model on Street Hazards"
)
parser.add_argument(
    "--with-anomalies",
    dest="with_anomalies",
    action="store_true",
    help="Include synthetic anomalies",
)
parser.add_argument(
    "--weights-path",
    dest="weights_path",
    type=str,
    default=None,
    help="Pretrained weights path",
)
parser.add_argument(
    "--config-path",
    dest="config_path",
    type=str,
    default="./config.yaml",
    help="YAML config file",
)
args = parser.parse_args()

# ── Environment configuration ────────────────────────────────────────────────
cv2.setNumThreads(0)
NUM_WORKERS: int = max(os.cpu_count() - 1, 1)

CONFIG_PATH = Path("./config.yaml")
config = load_config(str(CONFIG_PATH))
device = get_device()
fix_random(config["seed"])

TRAIN_ROOT = Path.home() / config["paths"]["train_path"]
VAL_ROOT = Path.home() / config["paths"]["test_path"]

common_loader_kwargs = dict(
    batch_size=config["training"]["batch_size"],
    num_workers=NUM_WORKERS,
    pin_memory=True,
    persistent_workers=True,
)


if __name__ == "__main__":
    # ── Data ──────────────────────────────────────────────────────────────────
    data_transforms = get_data_transforms(
        img_height=config["training"]["img_height"],
        img_width=config["training"]["img_width"],
    )
    denorm = Denormalize()

    num_classes = len(STREET_HAZARDS_CLASSES) - 1  # discard 'unknown'
    class_dict = get_classes_as_dict()

    train_dataset = StreetHazards(
        root_dir=TRAIN_ROOT,
        subset="training",
        transforms=data_transforms["train"],
        add_anomalies=args.with_anomalies,
    )

    val_dataset = StreetHazards(
        root_dir=TRAIN_ROOT,
        subset="validation",
        transforms=data_transforms["val"],
        add_anomalies=args.with_anomalies,
    )

    train_loader = MultiEpochsDataLoader(
        train_dataset,
        shuffle=True,
        drop_last=True,
        **common_loader_kwargs,
    )
    val_loader = MultiEpochsDataLoader(
        val_dataset,
        shuffle=False,
        drop_last=True,
        **common_loader_kwargs,
    )

    # ── Losses ───────────────────────────────────────────────────────────────
    triplet_w, focal_w = (0.45, 0.45) if args.with_anomalies else (0.5, 0.5)

    criterions = [
        (
            triplet_w,
            PrototypicalGlobalLocalTripletLoss(
                num_classes=num_classes,
                margin_global=config["losses"]["prototypical_triplet"]["global"],
                margin_local=config["losses"]["prototypical_triplet"]["local"],
                magnitude=config["losses"]["prototypical_triplet"]["anchors_magnitude"],
            ),
        ),
        (
            focal_w,
            FocalLoss(
                alpha=config["losses"]["focal"]["alpha"],
                gamma=config["losses"]["focal"]["gamma"],
                reduction="mean",
                ignore_index=13,
            ),
        ),
    ]
    if args.with_anomalies:
        criterions.append(
            (
                0.10,
                ObjectosphereLoss(
                    xi=config["losses"]["objectosphere"]["xi"],
                    unknown_label=13,
                ),
            )
        )

    # ── Model ────────────────────────────────────────────────────────────────
    model, model_info = build_model(config, num_classes)
    print_summary(
        model,
        (
            config["training"]["batch_size"],
            3,
            config["training"]["img_height"],
            config["training"]["img_width"],
        ),
    )

    if args.weights_path:
        weights_path = Path(args.weights_path)
        if weights_path.exists():
            state_dict = torch.load(
                weights_path, weights_only=True, map_location=device
            )
            model.load_state_dict(state_dict)
            logger.info("Loaded weights from {}", weights_path)
        else:
            logger.warning(
                "Weights path {} does not exist. Continuing without loading.",
                args.weights_path,
            )

    # ── Trainer ──────────────────────────────────────────────────────────────
    trainer = Trainer(
        config,
        model,
        device,
        train_loader,
        criterions,
        denorm,
        val_loader,
        class_dict,
        [MeanIoU(num_classes, ignore_index=13)],
        True,
    )

    run_name = generate_run_name(config, model_info, criterions)
    trainer.train(f"{run_name}")
