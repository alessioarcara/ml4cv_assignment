from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from loguru import logger
from prettytable import PrettyTable
from pydantic import validate_call

import wandb
from ml4cv_assignment.utils.typings import MetricModality


class WandBRetriever:
    def __init__(
        self,
        entity: str,
        project: str,
        download_dir: str = "./wandb_downloads",
    ):
        self.entity = entity
        self.project = project
        self.api = wandb.Api(overrides={"project": project, "entity": entity})
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)

    def _get_step_from_name(self, filename: str) -> int:
        """
        Extracts step from WandB filenames (e.g., 'media/images/pca_3229_123abc.png' -> 3229).
        """
        parts = Path(filename).stem.split("_")
        for part in parts:
            if part.isdigit():
                return int(part)
        return -1

    def _download_media(self, file_obj, run_id: str) -> Optional[str]:
        """
        Downloads a specified media file from a WandB run and saves it locally.
        """
        try:
            run_dir = self.download_dir / run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            # .download() returns a file object that needs to be closed after use
            f = file_obj.download(root=run_dir, replace=True)
            f.close()
            return str(run_dir / file_obj.name)
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
            # 1. Retrieve the run
            try:
                path = f"{self.entity}/{self.project}/{rid}"
                run = self.api.run(path)
            except Exception as e:
                logger.error(f"⚠️ Failed to retrieve run {rid}: {e}")
                continue

            run_result = {"run_name": run.name}

            # 2. Fetch history
            df = run.history(keys=keys_to_fetch, pandas=True)

            if reference_metric not in df.columns:
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

                        run_result[key] = self._download_media(target_file, run.name)

                    except IndexError:
                        logger.warning(
                            f"⚠️ No media file found for key '{key}' in run {run.name}."
                        )
                        run_result[key] = None

            data.append(run_result)

        return data
