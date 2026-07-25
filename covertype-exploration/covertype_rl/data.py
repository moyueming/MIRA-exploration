from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


CONTINUOUS_COLUMNS = [
    "Elevation",
    "Aspect",
    "Slope",
    "Horizontal_Distance_To_Hydrology",
    "Vertical_Distance_To_Hydrology",
    "Horizontal_Distance_To_Roadways",
    "Hillshade_9am",
    "Hillshade_Noon",
    "Hillshade_3pm",
    "Horizontal_Distance_To_Fire_Points",
]
TARGET_COLUMN = "Cover_Type"
WILDERNESS_COLUMNS = [f"Wilderness_Area{i}" for i in range(1, 5)]
SOIL_COLUMNS = [f"Soil_Type{i}" for i in range(1, 41)]


@dataclass
class CovertypeData:
    frame: pd.DataFrame
    continuous: np.ndarray
    continuous_norm: np.ndarray
    continuous_bins: np.ndarray
    cover_type: np.ndarray
    wilderness: np.ndarray
    soil: np.ndarray
    continuous_edges: list
    n_bins: int

    @property
    def n_rows(self):
        return int(self.continuous.shape[0])


def load_covertype(csv_path, n_bins=10):
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing Covertype CSV: {csv_path}")

    frame = pd.read_csv(csv_path)
    missing = [col for col in CONTINUOUS_COLUMNS + [TARGET_COLUMN] if col not in frame.columns]
    if missing:
        raise ValueError(f"Covertype CSV is missing required columns: {missing}")

    continuous = frame[CONTINUOUS_COLUMNS].to_numpy(dtype=np.float32)
    mean = continuous.mean(axis=0, keepdims=True)
    std = continuous.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    continuous_norm = ((continuous - mean) / std).astype(np.float32)

    bins = np.zeros_like(continuous, dtype=np.int8)
    edges_by_col = []
    for idx in range(continuous.shape[1]):
        edges = np.quantile(continuous[:, idx], np.linspace(0.0, 1.0, n_bins + 1)[1:-1])
        edges_by_col.append(edges.astype(np.float32))
        bins[:, idx] = np.searchsorted(edges, continuous[:, idx], side="right").astype(np.int8)

    wilderness = _one_hot_label(frame, WILDERNESS_COLUMNS)
    soil = _one_hot_label(frame, SOIL_COLUMNS)

    return CovertypeData(
        frame=frame,
        continuous=continuous,
        continuous_norm=continuous_norm,
        continuous_bins=bins,
        cover_type=frame[TARGET_COLUMN].to_numpy(dtype=np.int16),
        wilderness=wilderness,
        soil=soil,
        continuous_edges=edges_by_col,
        n_bins=int(n_bins),
    )


def _one_hot_label(frame, columns):
    available = [col for col in columns if col in frame.columns]
    if not available:
        return np.zeros(len(frame), dtype=np.int16)
    values = frame[available].to_numpy(dtype=np.int8)
    return (np.argmax(values, axis=1) + 1).astype(np.int16)
