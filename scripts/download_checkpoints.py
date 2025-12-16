#!/usr/bin/env -S uv run --script
#
# /// script
# dependencies = ["gdown", "loguru"]
# ///

from pathlib import Path

import gdown
from loguru import logger

CHECKPOINTS_DIR = Path("checkpoints")
CHECKPOINTS_DIR.mkdir(exist_ok=True)

# Dict: Filename -> Google Drive ID
files = {
    "ood_baseline_proto_segnet_20251213_052614_val-OoDAUPR_epoch_0.7577.pth": "15fM3p8VJRrLllwiO7Fu7f-O7N1LuGaqR",
    "ptl_dice_objectosphere_20251208_222812_val-OoDAUPR_epoch_0.7698.pth": "1WaaYNKvnsZiBZ98xj5zM-Nawzo78xDGH",
    "ood_mask2former_20251214_121509_val-OoDAUPR_epoch_0.6609.pth": "16Js6iKtZAV30Mj56064mOYSVloMTtAOp",
}


logger.info(f"Downloading models from Google Drive to '{CHECKPOINTS_DIR}'...")
for filename, file_id in files.items():
    url = f"https://drive.google.com/uc?id={file_id}"
    output_path = CHECKPOINTS_DIR / filename

    logger.info(f"Downloading {filename}...")
    gdown.download(url, str(output_path), quiet=False)

logger.success(f"Done! All models are located in '{CHECKPOINTS_DIR}'")
