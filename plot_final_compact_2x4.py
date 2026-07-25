from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from plot_compact_two_dataset_2x4_sample import (
    ALL_METHODS,
    DISPLAY_LABELS,
    MARKERS,
    METRICS,
    _configure_style,
    _draw_panel,
    _metric_limits,
)
from plot_final_cumulative_performance import load_metric_seed_curves
from plot_galaxy_extrinsic_reward_template import (
    ABLATION_METHODS,
    MAIN_METHODS,
    METHOD_STYLES,
)
from plot_real_episode_reward_two_datasets import DATASET_FILES


ROOT = Path(__file__).resolve().parent

CSV_METRICS = {
    "cumulative_reward": "cumulative_extrinsic_reward",
    "cumulative_target_efficiency": "cumulative_target_efficiency",
    "cumulative_unique_sets": "cumulative_unique_sets_viewed",
}

SMOOTHING_WINDOWS = {
    "Galaxy": 25,
    "Covertype": 50,
}

OUTPUT_FOLDERS = {
    "Galaxy": "galaxy_final",
    "Covertype": "covertype_final",
}

OUTPUT_NAMES = {
    "Galaxy": "galaxy_compact_2x4.png",
    "Covertype": "covertype_compact_2x4.png",
}


def _load_csv_metric(paths: tuple[Path, ...], metric: str) -> np.ndarray:
    curves = []
    for path in paths:
        frame = pd.read_csv(path, usecols=["episode", metric])
        frame["episode"] = pd.to_numeric(frame["episode"], errors="coerce")
        frame[metric] = pd.to_numeric(frame[metric], errors="coerce")
        frame = frame.dropna().sort_values("episode")
        frame["episode"] = frame["episode"].astype(int)
        frame = frame.drop_duplicates("episode", keep="last").set_index("episode")
        values = frame[metric].reindex(range(1, 1001))
        if values.isna().any():
            raise ValueError(f"{path} has incomplete {metric} values")
        curves.append(values.to_numpy(dtype=float))
    return np.vstack(curves)

def _smooth_per_seed(seed_curves: np.ndarray, window: int) -> np.ndarray:
    return np.vstack(
        [
            pd.Series(seed)
            .rolling(window=window, min_periods=1)
            .mean()
            .to_numpy()
            for seed in seed_curves
        ]
    )


def load_real_compact_data(
    dataset: str,
) -> dict[str, dict[str, np.ndarray]]:
    if dataset not in DATASET_FILES:
        raise ValueError(f"unsupported dataset: {dataset}")

    data: dict[str, dict[str, np.ndarray]] = {}
    for method in ALL_METHODS:
        paths = DATASET_FILES[dataset][method]
        method_data = {
            display_metric: load_metric_seed_curves(paths, csv_metric)
            for display_metric, csv_metric in CSV_METRICS.items()
        }
        method_data["target_efficiency"] = _smooth_per_seed(
            _load_csv_metric(paths, "target_efficiency"),
            SMOOTHING_WINDOWS[dataset],
        )
        data[method] = method_data
    return data


def _legend_handles() -> list[Line2D]:
    handles = []
    for method in ALL_METHODS:
        color, _, _ = METHOD_STYLES[method]
        handles.append(
            Line2D(
                [0],
                [0],
                color=color,
                linestyle="-",
                linewidth=1.8,
                marker=MARKERS[method],
                markersize=4.0,
                markerfacecolor="white",
                markeredgewidth=0.8,
            )
        )
    return handles


def build_final_compact_figure(
    dataset: str,
) -> tuple[plt.Figure, np.ndarray]:
    _configure_style()
    data = load_real_compact_data(dataset)
    fig, axes = plt.subplots(2, 4, figsize=(7.2, 4.3), squeeze=False)

    panel_index = 0
    for column, (metric, config) in enumerate(METRICS.items()):
        ylim = _metric_limits(data, metric)
        for row, methods in enumerate((MAIN_METHODS, ABLATION_METHODS)):
            ax = axes[row, column]
            panel_label = f"({chr(ord('a') + row * 4 + column)})"
            _draw_panel(ax, data, metric, methods, panel_label, ylim)
            ax.set_title(f"{panel_label} {config['title']}", pad=4.0)
            ax.set_xlabel("Episode", labelpad=2.0)

    legend = fig.legend(
        _legend_handles(),
        [DISPLAY_LABELS[method] for method in ALL_METHODS],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.008),
        ncol=4,
        frameon=False,
        handlelength=2.0,
        columnspacing=1.0,
        handletextpad=0.45,
        prop={"weight": "bold", "size": 7.1},
    )
    for handle in legend.legend_handles:
        handle.set_linewidth(2.0)

    fig.suptitle(dataset, y=0.995, fontsize=9.6, fontweight="bold")
    fig.subplots_adjust(
        top=0.89,
        bottom=0.20,
        left=0.055,
        right=0.985,
        hspace=0.55,
        wspace=0.30,
    )
    return fig, axes


def render_final_compact_pngs(final_root: Path) -> dict[str, Path]:
    outputs = {}
    for dataset in ("Galaxy", "Covertype"):
        folder = final_root / OUTPUT_FOLDERS[dataset]
        folder.mkdir(parents=True, exist_ok=True)
        figure, _ = build_final_compact_figure(dataset)
        output = folder / OUTPUT_NAMES[dataset]
        figure.savefig(output, dpi=300, facecolor="white")
        plt.close(figure)
        outputs[dataset] = output
    return outputs


if __name__ == "__main__":
    render_final_compact_pngs(ROOT / "outputs" / "final_results")
