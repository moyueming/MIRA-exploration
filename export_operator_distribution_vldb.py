from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


OPERATOR_KEYS = ("facet", "superset", "neighbor", "distribution")


def normalize_operator(value: object) -> str:
    text = str(value).strip().lower()
    for key in OPERATOR_KEYS:
        if key in text:
            return key
    raise ValueError(f"Unknown operator: {value!r}")


def operator_ratio_matrix(paths: list[Path | None]) -> np.ndarray:
    rows: list[np.ndarray] = []
    for path in paths:
        if path is None:
            continue
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path, usecols=["operator"])
        labels = frame["operator"].dropna().map(normalize_operator)
        counts = labels.value_counts()
        total = int(counts.sum())
        if total == 0:
            raise ValueError(f"No operators in {path}")
        rows.append(
            np.array(
                [counts.get(key, 0) / total for key in OPERATOR_KEYS],
                dtype=float,
            )
        )
    if not rows:
        raise ValueError("No exploration traces were available")
    return np.vstack(rows)
