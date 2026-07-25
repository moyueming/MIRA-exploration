import csv
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
BENCHMARK_DIR = ROOT_DIR / "ATENA-A-EDA" / "benchmark"
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

from .env import dataset_enum


def official_metrics(schema: str, dataset_number: int, actions: List[object]) -> Dict[str, float]:
    from atena.evaluation.metrics import EvalInstance, get_dataframe_all_eval_metrics

    schema_name, dataset_name = dataset_enum(schema, dataset_number)
    df = get_dataframe_all_eval_metrics([EvalInstance(_meta(schema_name, dataset_name), actions)])
    row = df.iloc[0].to_dict()
    return {str(key): float(value) for key, value in row.items()}


def _meta(schema_name, dataset_name):
    from atena.simulation.dataset import DatasetMeta

    return DatasetMeta(schema_name, dataset_name)


def summarize_metrics(rows: List[Dict[str, object]]) -> Dict[str, float]:
    metric_names = ["Precision", "T-BLEU-1", "T-BLEU-2", "T-BLEU-3", "EDA-Sim"]
    summary = {}
    for metric in metric_names:
        values = [float(row[metric]) for row in rows if metric in row]
        if values:
            summary[f"{metric}_mean"] = float(np.mean(values))
            summary[f"{metric}_std"] = float(np.std(values, ddof=0))
    return summary


def write_metrics_csv(path: Path, rows: List[Dict[str, object]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
