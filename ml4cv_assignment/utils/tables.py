from typing import Any, Dict, List, Literal

import numpy as np
from prettytable.colortable import ColorTable, Themes

from ml4cv_assignment.utils.constants import KEY_MIOU, KEY_OOD


def print_eval_results(results: Dict[str, float], title: str) -> None:
    table = ColorTable(theme=Themes.PASTEL)
    table.title = title
    table.field_names = ["Metric", "Value"]
    table.align["Metric"] = "l"
    table.align["Value"] = "r"

    for key, value in results.items():
        table.add_row([key, f"{value:.2f}"])

    print(table)


def print_metrics_table(
    data: List[Dict[str, Any]], mode: Literal["closed", "open"]
) -> None:
    table = ColorTable(theme=Themes.PASTEL)
    table.title = f"Best Validation Metrics ({mode.capitalize()} Set)"

    columns = [("Run", "l", "run_name"), ("mIoU", "r", KEY_MIOU)]
    if mode == "open":
        columns.append(("OoDAUPR", "r", KEY_OOD))

    table.field_names = [col[0] for col in columns]
    for name, align, _ in columns:
        table.align[name] = align  # type: ignore

    def get_max_val(run_data: dict, key: str) -> str:
        arr = np.array(run_data.get(key, []))
        val = np.max(arr) if arr.size > 0 else 0.0
        return f"{val:.3f}"

    for run in data:
        row = []
        row.append(run["run_name"])

        row.append(get_max_val(run, KEY_MIOU))

        if mode == "open":
            row.append(get_max_val(run, KEY_OOD))

        table.add_row(row)

    print(table)
