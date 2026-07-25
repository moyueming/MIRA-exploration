from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

from final_figure_registry import (
    ABLATION_METHODS,
    ALL_METHODS,
    DISPLAY_LABELS,
    MAIN_METHODS,
    METHOD_MARKERS as MARKERS,
    METHOD_STYLES,
)


ROOT = Path(__file__).resolve().parent
EPISODES = np.arange(1, 1001)

METRICS = {
    "cumulative_reward": {
        "title": "Cumulative Extrinsic Reward",
        "step": False,
    },
    "cumulative_target_efficiency": {
        "title": "Cumulative Target Efficiency",
        "step": False,
    },
    "cumulative_unique_sets": {
        "title": "Cumulative Unique Sets Viewed",
        "step": True,
    },
    "target_efficiency": {
        "title": "Target Efficiency",
        "step": False,
    },
}

QUALITY = {
    "MIRA": 1.00,
    "MIRA w/o Ext. Reward": 0.78,
    "DORA": 0.73,
    "Greedy": 0.64,
    "ATENA": 0.68,
    "ATENA w/o Ext. Reward": 0.53,
    "A3C": 0.59,
    "Random": 0.34,
}


def _correlated_noise(
    rng: np.random.Generator,
    scale: float,
    persistence: float = 0.92,
) -> np.ndarray:
    innovations = rng.normal(0.0, scale, size=EPISODES.size)
    values = np.empty(EPISODES.size, dtype=float)
    values[0] = innovations[0]
    for index in range(1, values.size):
        values[index] = persistence * values[index - 1] + innovations[index]
    return values * np.sqrt(1.0 - persistence**2)


def generate_synthetic_curves(
    dataset: str,
) -> dict[str, dict[str, np.ndarray]]:
    if dataset not in {"Galaxy", "Covertype"}:
        raise ValueError(f"unsupported dataset: {dataset}")

    dataset_index = 0 if dataset == "Galaxy" else 1
    reward_scale = 1.0 if dataset == "Galaxy" else 1.45
    set_scale = 1.0 if dataset == "Galaxy" else 1.25
    noise_scale = 1.0 if dataset == "Galaxy" else 1.45
    progress = 0.38 * (1.0 - np.exp(-EPISODES / 220.0))
    progress += 0.62 / (1.0 + np.exp(-(EPISODES - 620.0) / 125.0))
    progress /= progress[-1]

    result: dict[str, dict[str, np.ndarray]] = {}
    for method_index, method in enumerate(ALL_METHODS):
        quality = QUALITY[method]
        metric_seeds = {metric: [] for metric in METRICS}
        for seed in range(3):
            rng = np.random.default_rng(
                31_000 + dataset_index * 10_000 + method_index * 100 + seed
            )
            seed_factor = 1.0 + rng.normal(0.0, 0.045 * noise_scale)
            reward_trend = reward_scale * (
                4.0 + 8.0 * quality + (5.0 + 16.0 * quality) * progress
            )
            raw_reward = reward_trend * seed_factor
            raw_reward += _correlated_noise(rng, 1.3 * noise_scale)
            raw_reward = np.maximum(raw_reward, 0.15)
            episode_reward = (
                pd.Series(raw_reward)
                .rolling(window=25 if dataset == "Galaxy" else 40, min_periods=1)
                .mean()
                .to_numpy()
            )

            reward_increment = np.maximum(raw_reward + rng.normal(0.0, 0.8, 1000), 0.0)
            cumulative_reward = np.cumsum(reward_increment)

            discovery_rate = set_scale * (
                1.0 + 4.2 * quality - (0.30 + 0.55 * quality) * progress
            )
            discoveries = rng.poisson(np.maximum(discovery_rate, 0.10)).astype(float)
            cumulative_unique_sets = np.cumsum(discoveries)

            efficiency = 0.05 + 0.34 * quality * (1.0 - np.exp(-EPISODES / 260.0))
            efficiency += _correlated_noise(rng, 0.012 * noise_scale, persistence=0.96)
            efficiency = np.clip(efficiency * seed_factor, 0.0, None)
            cumulative_efficiency = np.cumsum(efficiency) / EPISODES
            target_efficiency = (
                pd.Series(efficiency)
                .rolling(window=25 if dataset == "Galaxy" else 50, min_periods=1)
                .mean()
                .to_numpy()
            )

            metric_seeds["cumulative_reward"].append(cumulative_reward)
            metric_seeds["cumulative_target_efficiency"].append(
                cumulative_efficiency
            )
            metric_seeds["cumulative_unique_sets"].append(cumulative_unique_sets)
            metric_seeds["target_efficiency"].append(target_efficiency)

        result[method] = {
            metric: np.vstack(seed_values)
            for metric, seed_values in metric_seeds.items()
        }
    return result


def _configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 7.4,
            "font.weight": "bold",
            "axes.labelsize": 8.0,
            "axes.labelweight": "bold",
            "axes.titlesize": 8.4,
            "axes.titleweight": "bold",
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 6.8,
            "legend.fontsize": 7.1,
            "axes.linewidth": 0.8,
            "lines.solid_capstyle": "round",
        }
    )


def _statistics(seed_curves: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return seed_curves.mean(axis=0), seed_curves.std(axis=0, ddof=1)


def _metric_limits(
    data: dict[str, dict[str, np.ndarray]],
    metric: str,
) -> tuple[float, float]:
    lower_bounds = []
    upper_bounds = []
    for method in ALL_METHODS:
        mean, sd = _statistics(data[method][metric])
        lower_bounds.append(float(np.min(mean - sd)))
        upper_bounds.append(float(np.max(mean + sd)))
    lower = min(lower_bounds)
    upper = max(upper_bounds)
    padding = 0.045 * max(upper - lower, 1.0)
    return min(0.0, lower - padding), upper + padding


def _draw_panel(
    ax: plt.Axes,
    data: dict[str, dict[str, np.ndarray]],
    metric: str,
    methods: tuple[str, ...],
    panel_label: str,
    ylim: tuple[float, float],
) -> None:
    is_step = bool(METRICS[metric]["step"])
    for method in methods:
        mean, sd = _statistics(data[method][metric])
        color, _, linewidth = METHOD_STYLES[method]
        fill_options = {"step": "post"} if is_step else {}
        ax.fill_between(
            EPISODES,
            mean - sd,
            mean + sd,
            color=color,
            alpha=0.10,
            linewidth=0,
            zorder=1,
            **fill_options,
        )
        ax.plot(
            EPISODES,
            mean,
            color=color,
            linestyle="-",
            linewidth=max(0.85, linewidth * 0.82),
            drawstyle="steps-post" if is_step else "default",
            marker=MARKERS[method],
            markevery=[199, 399, 599, 799, 999],
            markersize=2.8,
            markerfacecolor="white",
            markeredgewidth=0.65,
            label=method,
            zorder=3 if method == "MIRA" else 2,
        )

    ax.set_xlim(0, 1000)
    ax.set_ylim(*ylim)
    ax.set_xticks((0, 250, 500, 750, 1000))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.grid(axis="y", color="#D0D0D0", linewidth=0.38, alpha=0.62)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(0.8)
    ax.tick_params(direction="out", length=2.4, width=0.75, pad=1.5)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontweight("bold")


def build_compact_sample_figure(
    dataset: str,
) -> tuple[plt.Figure, np.ndarray]:
    _configure_style()
    data = generate_synthetic_curves(dataset)
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
    legend = fig.legend(
        handles,
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


def render_compact_samples(output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for dataset in ("Galaxy", "Covertype"):
        figure, _ = build_compact_sample_figure(dataset)
        output = output_dir / f"{dataset.lower()}_compact_2x4_sample.png"
        figure.savefig(output, dpi=300, facecolor="white")
        plt.close(figure)
        outputs[dataset] = output
    return outputs


if __name__ == "__main__":
    render_compact_samples(ROOT / "outputs" / "figure_templates")
