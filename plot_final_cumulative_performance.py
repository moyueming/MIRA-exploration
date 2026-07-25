from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from final_figure_registry import ALL_METHODS

from plot_galaxy_extrinsic_reward_template import (
    ABLATION_METHODS,
    MAIN_METHODS,
    METHOD_STYLES,
)
from plot_real_episode_reward_two_datasets import DATASET_FILES


ROOT = Path(__file__).resolve().parent
MAX_EPISODE = 1000

METRICS = {
    "cumulative_extrinsic_reward": {
        "stem": "cumulative_reward",
        "ylabel": "Cumulative Extrinsic Reward",
        "step": None,
    },
    "cumulative_target_efficiency": {
        "stem": "cumulative_target_efficiency",
        "ylabel": "Cumulative Target Efficiency",
        "step": None,
    },
    "cumulative_unique_sets_viewed": {
        "stem": "cumulative_unique_sets",
        "ylabel": "Cumulative Unique Sets Viewed",
        "step": "post",
    },
}

VARIANTS = {
    "main": MAIN_METHODS,
    "ablation": ABLATION_METHODS,
    "all_methods": ALL_METHODS,
}

OUTPUT_FOLDERS = {
    "Galaxy": "galaxy_final",
    "Covertype": "covertype_final",
}

OUTPUT_NAMES = {
    dataset: {
        (metric, variant): (
            f"{dataset.lower()}_{config['stem']}_{variant}.png"
        )
        for metric, config in METRICS.items()
        for variant in VARIANTS
    }
    for dataset in ("Galaxy", "Covertype")
}


def load_metric_seed_curves(paths: tuple[Path, ...], metric: str) -> np.ndarray:
    if metric not in METRICS:
        raise ValueError(f"unsupported cumulative metric: {metric}")
    if len(paths) != 3:
        raise ValueError("exactly three seed paths are required")

    seed_curves = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path, usecols=lambda column: column in {"episode", metric})
        missing = {"episode", metric}.difference(frame.columns)
        if missing:
            raise KeyError(f"{path} is missing columns: {sorted(missing)}")

        frame["episode"] = pd.to_numeric(frame["episode"], errors="coerce")
        frame[metric] = pd.to_numeric(frame[metric], errors="coerce")
        frame = frame.dropna().sort_values("episode")
        frame["episode"] = frame["episode"].astype(int)
        frame = frame.drop_duplicates("episode", keep="last").set_index("episode")
        values = frame[metric].reindex(range(1, MAX_EPISODE + 1))
        if values.isna().any():
            missing_episodes = values.index[values.isna()].tolist()
            raise ValueError(f"{path} is missing episodes: {missing_episodes[:10]}")
        seed_curves.append(values.to_numpy(dtype=float))
    return np.vstack(seed_curves)


def metric_statistics(seed_curves: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(seed_curves, dtype=float)
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError(
            "seed_curves must have shape (n_seeds, n_episodes) with n_seeds >= 2"
        )
    return np.mean(values, axis=0), np.std(values, axis=0, ddof=1)


def compute_metric_statistics(
    dataset: str,
    metric: str,
    methods: tuple[str, ...],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    if dataset not in DATASET_FILES:
        raise ValueError(f"unsupported dataset: {dataset}")
    files = DATASET_FILES[dataset]
    return {
        method: metric_statistics(load_metric_seed_curves(files[method], metric))
        for method in methods
    }


def _configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 10.5,
            "font.weight": "bold",
            "axes.labelsize": 12,
            "axes.labelweight": "bold",
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 9.5,
            "axes.linewidth": 0.9,
            "lines.solid_capstyle": "round",
        }
    )


def _limits(
    statistics: dict[str, tuple[np.ndarray, np.ndarray]],
) -> tuple[float, float]:
    lower = min(float(np.min(mean - sd)) for mean, sd in statistics.values())
    upper = max(float(np.max(mean + sd)) for mean, sd in statistics.values())
    padding = 0.04 * max(upper - lower, 1.0)
    return min(0.0, lower - padding), upper + padding


def _bold_legend(legend) -> None:
    for text in legend.get_texts():
        text.set_fontweight("bold")
    for handle in legend.legend_handles:
        handle.set_linewidth(2.8)
        handle.set_linestyle("-")


def build_metric_figure(
    dataset: str,
    metric: str,
    methods: tuple[str, ...],
) -> tuple[plt.Figure, plt.Axes]:
    if metric not in METRICS:
        raise ValueError(f"unsupported cumulative metric: {metric}")
    _configure_style()
    statistics = compute_metric_statistics(dataset, metric, methods)
    config = METRICS[metric]
    episodes = np.arange(1, MAX_EPISODE + 1)
    fig, ax = plt.subplots(figsize=(7.6, 3.9))

    for method in methods:
        mean, sd = statistics[method]
        color, _, linewidth = METHOD_STYLES[method]
        fill_options = {"step": config["step"]} if config["step"] else {}
        ax.fill_between(
            episodes,
            mean - sd,
            mean + sd,
            color=color,
            alpha=0.12,
            linewidth=0,
            zorder=1,
            **fill_options,
        )
        ax.plot(
            episodes,
            mean,
            color=color,
            linestyle="-",
            linewidth=linewidth,
            drawstyle="steps-post" if config["step"] else "default",
            label=method,
            zorder=3 if method == "MIRA" else 2,
        )

    ax.set_xlim(0, MAX_EPISODE)
    ax.set_ylim(*_limits(statistics))
    ax.set_title(dataset, pad=7)
    ax.set_xlabel("Episode")
    ax.set_ylabel(config["ylabel"])
    ax.grid(axis="y", color="#D0D0D0", linewidth=0.5, alpha=0.70)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(0.9)
    ax.tick_params(direction="out", length=3.2, width=0.9, colors="black")
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")

    all_methods = len(methods) == len(ALL_METHODS)
    legend = fig.legend(
        *ax.get_legend_handles_labels(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=4 if all_methods else min(3, len(methods)),
        frameon=False,
        handlelength=2.5,
        columnspacing=1.0,
        handletextpad=0.5,
    )
    _bold_legend(legend)
    fig.subplots_adjust(
        top=0.70 if all_methods else 0.78,
        bottom=0.19,
        left=0.13,
        right=0.97,
    )
    return fig, ax


def _save_png(fig: plt.Figure, path: Path) -> Path:
    fig.savefig(path, dpi=300, facecolor="white")
    plt.close(fig)
    return path


def render_final_cumulative_pngs(
    final_root: Path,
) -> dict[str, dict[tuple[str, str], Path]]:
    outputs: dict[str, dict[tuple[str, str], Path]] = {}
    for dataset in ("Galaxy", "Covertype"):
        folder = final_root / OUTPUT_FOLDERS[dataset]
        folder.mkdir(parents=True, exist_ok=True)
        outputs[dataset] = {}
        for metric in METRICS:
            for variant, methods in VARIANTS.items():
                figure, _ = build_metric_figure(dataset, metric, methods)
                path = folder / OUTPUT_NAMES[dataset][(metric, variant)]
                outputs[dataset][(metric, variant)] = _save_png(figure, path)
    return outputs

def render_main_containing_cumulative_pngs(
    final_root: Path,
) -> dict[str, dict[tuple[str, str], Path]]:
    outputs: dict[str, dict[tuple[str, str], Path]] = {}
    for dataset in ("Galaxy", "Covertype"):
        folder = final_root / OUTPUT_FOLDERS[dataset]
        folder.mkdir(parents=True, exist_ok=True)
        outputs[dataset] = {}
        for metric in METRICS:
            for variant in ("main", "all_methods"):
                figure, _ = build_metric_figure(
                    dataset,
                    metric,
                    VARIANTS[variant],
                )
                path = folder / OUTPUT_NAMES[dataset][(metric, variant)]
                outputs[dataset][(metric, variant)] = _save_png(figure, path)
    return outputs

if __name__ == "__main__":
    render_final_cumulative_pngs(ROOT / "outputs" / "final_results")
