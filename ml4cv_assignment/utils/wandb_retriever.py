from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import pandas as pd
from loguru import logger
from prettytable import PrettyTable
from pydantic import validate_call

import wandb
from ml4cv_assignment.utils.typings import MetricModality


class WandBRetriever:
    """
    An utility class to retrieve metrics and media from WandB runs.
    """

    def __init__(
        self,
        entity: str,
        project: str,
        download_dir: str = "./wandb_downloads",
    ):
        self.entity = entity
        self.project = project
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)

        self._api: Optional[wandb.Api] = None

    @property
    def api(self):
        if self._api is None:
            logger.info("🔌 Connecting to WandB API...")
            self._api = wandb.Api(
                overrides={"project": self.project, "entity": self.entity}
            )
        return self._api

    def _get_step_from_name(self, filename: str) -> int:
        """
        Extracts step from WandB filenames (e.g., 'media/images/pca_3229_123abc.png' -> 3229).
        """
        parts = Path(filename).stem.split("_")
        for part in parts:
            if part.isdigit():
                return int(part)
        return -1

    def _download_media(self, file_obj, run_dir: Path) -> Optional[str]:
        """
        Downloads a specified media file from a WandB run and saves it locally.
        """
        target_path = run_dir / file_obj.name

        if target_path.exists():
            return str(target_path)

        try:
            # .download() returns a file object that needs to be closed after use
            f = file_obj.download(root=run_dir, replace=True)
            f.close()
            return str(target_path)
        except Exception:
            return None

    def show_available_runs(self) -> None:
        """
        Lists all runs available in the specified WandB project.
        """
        runs = self.api.runs(f"{self.entity}/{self.project}")

        table = PrettyTable()
        table.field_names = ["Run ID", "Name"]
        table.align = "l"  # type: ignore

        for run in runs:
            table.add_row(
                [
                    run.id,
                    run.name,
                ]
            )

        print(table)

    @validate_call
    def get_metrics(
        self,
        run_ids: List[str],
        metrics_config: Dict[str, MetricModality],
        reference_metric: str,
        mode: Literal["min", "max", "last"],
    ) -> List[Dict[str, Any]]:
        data = []

        scalar_keys = [
            k
            for k, t in metrics_config.items()
            if t in (MetricModality.FULL, MetricModality.SINGLE)
        ]
        media_keys = [k for k, t in metrics_config.items() if t == MetricModality.MEDIA]

        keys_to_fetch = list(set(scalar_keys + [reference_metric, "_step"]))

        for rid in run_ids:
            run_result: Dict[str, Any] = {}
            df = None

            run_dir = self.download_dir / rid
            run_dir.mkdir(parents=True, exist_ok=True)
            csv_path = run_dir / "history.csv"

            # 1. Try to load from cache first
            if csv_path.exists():
                logger.info(f"📂 Loading run {rid} from local CSV...")
                try:
                    df = pd.read_csv(csv_path)
                    run_result["run_name"] = str(df.iloc[0]["run_name"])
                except Exception as e:
                    logger.warning(f"⚠️ Failed to load local CSV for run {rid}: {e}")
                    df = None

            # 2. Retrieve from WandB if local not available
            # and cache it locally
            if df is None:
                try:
                    path = f"{self.entity}/{self.project}/{rid}"
                    run = self.api.run(path)

                    df = run.history(keys=keys_to_fetch, pandas=True)
                    df["run_name"] = run.name
                    run_result["run_name"] = run.name

                    df.to_csv(csv_path, index=False)

                except Exception as e:
                    logger.error(f"⚠️ Failed to retrieve run {rid}: {e}")
                    continue

            if df is None or reference_metric not in df.columns:
                logger.warning(
                    f"⚠️ Reference metric '{reference_metric}' not found in run {run.name}. Skipping."
                )
                data.append(run_result)
                continue

            # 3. Get the best index based on the reference metric
            ref_series = df[reference_metric]

            if mode == "min":
                best_idx = ref_series.idxmin()
            elif mode == "max":
                best_idx = ref_series.idxmax()
            else:  # mode == "last"
                best_idx = ref_series.index[-1]

            # 4. Collect scalar metrics
            for key in scalar_keys:
                if key not in df:
                    run_result[key] = None
                    continue

                modality = metrics_config[key]
                match modality:
                    case MetricModality.FULL:
                        run_result[key] = df[key].tolist()
                    case MetricModality.SINGLE:
                        run_result[key] = df.loc[best_idx, key]

            # 4. Download media files
            if media_keys:
                all_files = list(run.files())  # Costly API call, do it once

                for key in media_keys:
                    target_files = [f for f in all_files if key in f.name]

                    try:
                        target_file = target_files[best_idx]

                        run_result[key] = self._download_media(target_file, run_dir)

                    except IndexError:
                        logger.warning(
                            f"⚠️ No media file found for key '{key}' in run {run.name}."
                        )
                        run_result[key] = None

            data.append(run_result)

        return data
