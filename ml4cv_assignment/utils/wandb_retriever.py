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
        force_download: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Retrieves specified metrics and media from WandB runs with local caching.
        """
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

            # --- 1. Load Run History (Local Cache or WandB Download) ---
            # A. Attempt to load from local cache
            if csv_path.exists() and not force_download:
                try:
                    logger.info(f"📂 Loading run {rid} from local CSV...")
                    df = pd.read_csv(csv_path)
                    run_result["run_name"] = str(df.iloc[0]["run_name"])
                except Exception as e:
                    logger.warning(
                        f"⚠️ [Run {rid}] Failed to read local CSV, will re-download: {e}"
                    )
                    df = None

            # B. Download from WandB if cache is missing or forced
            if df is None:
                try:
                    logger.info(f"🌐 Retrieving run {rid} from WandB...")
                    run = self.api.run(f"{self.entity}/{self.project}/{rid}")

                    df = run.history(keys=keys_to_fetch, pandas=True, samples=100000)
                    df["run_name"] = run.name
                    run_result["run_name"] = run.name

                    df.to_csv(csv_path, index=False)
                except Exception as e:
                    logger.error(f"❌ [Run {rid}] Failed to retrieve run data: {e}")
                    continue

            if df is None or reference_metric not in df.columns:
                logger.warning(
                    f"⚠️ [Run {rid}] Reference metric '{reference_metric}' not found. Skipping."
                )
                data.append(run_result)
                continue

            # --- 2. Get the best index based on the reference metric ---
            if mode == "min":
                best_idx = df[reference_metric].idxmin()
            elif mode == "max":
                best_idx = df[reference_metric].idxmax()
            else:  # mode == "last"
                best_idx = df.index[-1]

            # --- 3. Collect scalar metrics ---
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

            # --- 4. Collect media files ---
            csv_updated = False
            wandb_files = None

            for key in media_keys:
                # A. Cache Hit: Check if the file path is already saved in the CSV and exists on disk.
                if key in df and pd.notna(df.loc[best_idx, key]):
                    local_p = Path(str(df.loc[best_idx, key]))
                    if local_p.exists():
                        run_result[key] = str(local_p)
                        continue

                # B. Cache Miss: We need to fetch from WandB.
                if wandb_files is None:
                    logger.info(
                        f"🔌 [Run {rid}] Connecting to API to fetch file list..."
                    )
                    try:
                        run_api = self.api.run(f"{self.entity}/{self.project}/{rid}")
                        wandb_files = list(run_api.files())
                    except Exception as e:
                        logger.error(f"❌ [Run {rid}] Failed to fetch file list: {e}")
                        wandb_files = []

                # Filter files matching the key (e.g., "pixel_embeddings_pca")
                target_files = [f for f in wandb_files if key in f.name]

                try:
                    # Select the file corresponding to the 'best_idx'.
                    target_file = target_files[best_idx]

                    # Download the file (function handles the physical disk check)
                    saved_path = self._download_media(target_file, run_dir)

                    if saved_path:
                        run_result[key] = saved_path

                        # Update DataFrame with the local path for future runs
                        if key not in df:
                            df[key] = None
                        df.loc[best_idx, key] = saved_path
                        csv_updated = True
                    else:
                        run_result[key] = None

                except IndexError:
                    logger.warning(
                        f"⚠️ [Run {rid}] Media '{key}' not found at index {best_idx}."
                    )
                    run_result[key] = None
                except Exception as e:
                    logger.error(
                        f"❌ [Run {rid}] Unexpected error processing '{key}': {e}"
                    )
                    run_result[key] = None

            if csv_updated:
                df.to_csv(csv_path, index=False)

            data.append(run_result)

        return data
