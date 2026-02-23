import argparse
import os
import subprocess
import sys
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
from loguru import logger
from tqdm import tqdm


def ensure_benchmark_code():
    current_file_path = Path(__file__).resolve()
    project_root = current_file_path.parent.parent
    external_dir = project_root / "external"
    target_path = external_dir / "road-anomaly-benchmark"

    REPO_URL = "https://github.com/SegmentMeIfYouCan/road-anomaly-benchmark.git"

    if target_path.exists() and (target_path.iterdir()):
        return

    logger.warning(
        "'road-anomaly-benchmark' not found in '{}'. Cloning from {}...",
        external_dir,
        REPO_URL,
    )

    try:
        external_dir.mkdir(parents=True, exist_ok=True)

        if target_path.exists():
            target_path.rmdir()

        subprocess.run(
            ["git", "clone", REPO_URL, str(target_path)],
            check=True,
            stdout=subprocess.DEVNULL,
        )
        logger.success("'road-anomaly-benchmark' cloned successfully.")
    except Exception as e:
        logger.error("Failed to clone 'road-anomaly-benchmark': {}", e)
        logger.error("Please manually clone it: git clone {} {}", REPO_URL, target_path)
        sys.exit(1)


def setup_environment():
    ensure_benchmark_code()
    current_file_path = Path(__file__).resolve()
    project_root = current_file_path.parent.parent
    external_repo_path = project_root / "external" / "road-anomaly-benchmark"
    sys.path.append(str(project_root))
    sys.path.append(str(external_repo_path))
    DATASET_ROOT = project_root / "datasets"
    os.environ["DIR_DATASETS"] = str(DATASET_ROOT)


setup_environment()

try:
    from road_anomaly_benchmark.evaluation import Evaluation  # type: ignore

    from ml4cv_assignment.config.utils import build_config
    from ml4cv_assignment.utils.misc import resolve_device
except ImportError:
    sys.exit(1)

# --- Config ---
CHECKPOINT_PATH = (
    "checkpoints/cityscapes_open_20251224_135733_val-OoDAUPR_epoch_0.7993.pth"
)
CONFIG_FILES = [
    "configs/base.yaml",
    "configs/transforms/ablation_result.yaml",
    "configs/decoders/fpn_no_aspp.yaml",
    "configs/losses/closed/ptl_focal_dice.yaml",
    "configs/dataset/cityscapes.yaml",
    "configs/dataset/cityscapes_open.yaml",
]


class AnomalyMethod:
    def __init__(
        self,
        checkpoint_path: str,
        config_files: list[str],
        scales: list[float] = [0.75, 1.0, 1.25],
        base_size: tuple[int, int] = (512, 1024),
    ):
        self.device = resolve_device()
        self.model = self._load_model(checkpoint_path, config_files)
        self.scales = scales
        self.base_size = base_size

        self.transform = A.Compose(
            [
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]
        )

    def _load_model(self, checkpoint_path: str, config_files: list[str]) -> nn.Module:
        cfg, _ = build_config(
            config_files,
            overrides={
                "model_cfg": {"model": {"params": {"alpha": 0.5}}},
                "paths": {"checkpoint": checkpoint_path},
            },
        )
        model = cfg.model
        model.eval()
        return model.to(self.device)

    @torch.inference_mode()
    def predict(self, image_np: np.ndarray) -> np.ndarray:
        original_h, original_w = image_np.shape[:2]

        processed = self.transform(image=image_np)["image"]
        input_tensor = processed.to(self.device).unsqueeze(0)

        total_ood_score = torch.zeros(
            (1, 1, original_h, original_w), device=self.device
        )
        count = 0

        # TTA: Multi-Scale + Flip
        for scale in self.scales:
            h_scaled = int(self.base_size[0] * scale)
            w_scaled = int(self.base_size[1] * scale)

            input_scaled = F.interpolate(
                input_tensor,
                size=(h_scaled, w_scaled),
                mode="bilinear",
                align_corners=False,
            )

            for do_flip in [False, True]:
                # Flip Input
                curr_input = (
                    torch.flip(input_scaled, dims=[3]) if do_flip else input_scaled
                )

                outputs = self.model({"pixel_values": curr_input})
                score = outputs["ood_score"]

                if score.dim() == 3:
                    score = score.unsqueeze(1)

                # Un-flip Output
                if do_flip:
                    score = torch.flip(score, dims=[3])

                score_up = F.interpolate(
                    score,
                    size=(original_h, original_w),
                    mode="bilinear",
                    align_corners=False,
                )
                total_ood_score += score_up
                count += 1

        avg_ood_score = total_ood_score / count

        ood_score_np = avg_ood_score.squeeze().cpu().numpy()

        ood_score_np = self._post_process(ood_score_np)

        return ood_score_np

    def _post_process(self, ood_score_np: np.ndarray) -> np.ndarray:
        return cv2.GaussianBlur(ood_score_np, (0, 0), sigmaX=20.0)


def main(args: argparse.Namespace):
    method = AnomalyMethod(checkpoint_path=CHECKPOINT_PATH, config_files=CONFIG_FILES)

    ev = Evaluation(
        method_name="ProtoSegNet",
        dataset_name=args.dataset,
    )

    logger.info("Starting benchmark on {} dataset...", args.dataset)
    for frame in tqdm(ev.get_frames(), desc="Processing", unit="frame"):
        anomaly_map = method.predict(frame.image)
        ev.save_output(frame, anomaly_map)

    ev.wait_to_finish_saving()
    logger.success("Benchmark completed.")
    try:
        metrics = ev.calculate_metric_from_saved_outputs("PixBinaryClass")
        print("\n--- Final Evaluation Metrics ---")
        print(metrics)
    except Exception:
        logger.warning(
            "Could not compute metrics. No annotations available for this dataset."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ProtoSegNet Benchmark Runner")
    parser.add_argument(
        "--dataset",
        type=str,
        default="AnomalyTrack-all",
        help="Dataset name in benchmark",
    )
    args = parser.parse_args()
    main(args)
