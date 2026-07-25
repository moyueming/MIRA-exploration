from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from plot_galaxy_extrinsic_reward_template import (
    ABLATION_METHODS,
    MAIN_METHODS,
    METHOD_STYLES,
    aggregate_seed_curves,
)
from plot_real_episode_reward_two_datasets import (
    DATASET_FILES,
    SMOOTHING_WINDOWS,
    load_seed_curves,
)


ROOT = Path(__file__).resolve().parent
MAX_EPISODE = 1000

OUTPUT_NAMES = {
    "Galaxy": {
        "main": "galaxy_episode_reward_main.png",
        "ablation": "galaxy_episode_reward_ablation.png",
        "combined": "galaxy_episode_reward_combined.png",
    },
    "Covertype": {
        "main": "covertype_episode_reward_main.png",
        "ablation": "covertype_episode_reward_ablation.png",
        "combined": "covertype_episode_reward_combined.png",
    },
}

OUTPUT_FOLDERS = {
    "Galaxy": "galaxy_final",
    "Covertype": "covertype_final",
}


def _configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 10.5,
            "axes.labelsize": 12,
            "axes.titlesize": 12,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 9.5,
            "axes.linewidth": 0.9,
        }
    )


def _statistics(
    dataset: str,
    methods: tuple[str, ...],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    files = DATASET_FILES[dataset]
    return {
        method: aggregate_seed_curves(
            load_seed_curves(files[method]),
            window=SMOOTHING_WINDOWS[dataset],
        )
        for method in methods
    }


def _limits(
    statistics: dict[str, tuple[np.ndarray, np.ndarray]],
) -> tuple[float, float]:
    lower = min(float(np.min(mean - sd)) for mean, sd in statistics.values())
    upper = max(float(np.max(mean + sd)) for mean, sd in statistics.values())
    padding = 0.04 * max(upper - lower, 1.0)
    return min(0.0, lower - padding), upper + padding


def _draw_panel(
    ax: plt.Axes,
    methods: tuple[str, ...],
    statistics: dict[str, tuple[np.ndarray, np.ndarray]],
    ylim: tuple[float, float],
    title: str,
) -> tuple[list, list[str]]:
    episodes = np.arange(1, MAX_EPISODE + 1)
    for method in methods:
        mean, sd = statistics[method]
        color, linestyle, linewidth = METHOD_STYLES[method]
        ax.fill_between(
            episodes,
            mean - sd,
            mean + sd,
            color=color,
            alpha=0.12,
            linewidth=0,
            zorder=1,
        )
        ax.plot(
            episodes,
            mean,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            label=method,
            zorder=3 if method == "MIRA" else 2,
        )

    ax.set_xlim(0, MAX_EPISODE)
    ax.set_ylim(*ylim)
    ax.set_title(title, fontweight="bold", pad=7)
    ax.set_xlabel("Episode", fontweight="bold")
    ax.set_ylabel("Episode Reward", fontweight="bold")
    ax.grid(axis="y", color="#D0D0D0", linewidth=0.5, alpha=0.70)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(0.9)
    ax.tick_params(direction="out", length=3.2, width=0.9, colors="black")
    return ax.get_legend_handles_labels()


def _bold_legend(legend) -> None:
    for text in legend.get_texts():
        text.set_fontweight("bold")
    for handle in legend.legend_handles:
        handle.set_linewidth(2.8)
        handle.set_linestyle("-")


def build_individual_figure(
    dataset: str,
    methods: tuple[str, ...],
) -> tuple[plt.Figure, plt.Axes]:
    _configure_style()
    statistics = _statistics(dataset, methods)
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    handles, labels = _draw_panel(
        ax,
        methods,
        statistics,
        _limits(statistics),
        dataset,
    )
    legend = fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=min(3, len(methods)),
        frameon=False,
        handlelength=2.5,
        columnspacing=1.1,
        handletextpad=0.5,
    )
    _bold_legend(legend)
    fig.subplots_adjust(top=0.79, bottom=0.20, left=0.11, right=0.96)
    return fig, ax


def build_combined_figure(dataset: str) -> tuple[plt.Figure, np.ndarray]:
    _configure_style()
    all_methods = tuple(dict.fromkeys(MAIN_METHODS + ABLATION_METHODS))
    statistics = _statistics(dataset, all_methods)
    shared_ylim = _limits(statistics)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), squeeze=False)
    axes = axes[0]

    left_handles, left_labels = _draw_panel(
        axes[0],
        MAIN_METHODS,
        statistics,
        shared_ylim,
        "(a) Main comparison",
    )
    right_handles, right_labels = _draw_panel(
        axes[1],
        ABLATION_METHODS,
        statistics,
        shared_ylim,
        "(b) Ablation",
    )

    left_legend = fig.legend(
        left_handles,
        left_labels,
        loc="upper center",
        bbox_to_anchor=(0.27, 0.995),
        ncol=3,
        frameon=False,
        handlelength=2.4,
        columnspacing=0.9,
        handletextpad=0.45,
    )
    right_legend = fig.legend(
        right_handles,
        right_labels,
        loc="upper center",
        bbox_to_anchor=(0.76, 0.995),
        ncol=2,
        frameon=False,
        handlelength=2.4,
        columnspacing=0.9,
        handletextpad=0.45,
    )
    _bold_legend(left_legend)
    _bold_legend(right_legend)
    fig.subplots_adjust(top=0.76, bottom=0.19, left=0.075, right=0.975, wspace=0.24)
    return fig, axes


def _save_png(fig: plt.Figure, path: Path) -> Path:
    fig.savefig(path, dpi=300, facecolor="white")
    plt.close(fig)
    return path


def render_final_pngs(final_root: Path) -> dict[str, dict[str, Path]]:
    outputs: dict[str, dict[str, Path]] = {}
    for dataset in ("Galaxy", "Covertype"):
        folder = final_root / OUTPUT_FOLDERS[dataset]
        folder.mkdir(parents=True, exist_ok=True)

        main_figure, _ = build_individual_figure(dataset, MAIN_METHODS)
        ablation_figure, _ = build_individual_figure(dataset, ABLATION_METHODS)
        combined_figure, _ = build_combined_figure(dataset)
        names = OUTPUT_NAMES[dataset]
        outputs[dataset] = {
            "main": _save_png(main_figure, folder / names["main"]),
            "ablation": _save_png(ablation_figure, folder / names["ablation"]),
            "combined": _save_png(combined_figure, folder / names["combined"]),
        }
    return outputs

def render_main_containing_episode_pngs(
    final_root: Path,
) -> dict[str, dict[str, Path]]:
    outputs: dict[str, dict[str, Path]] = {}
    for dataset in ("Galaxy", "Covertype"):
        folder = final_root / OUTPUT_FOLDERS[dataset]
        folder.mkdir(parents=True, exist_ok=True)
        names = OUTPUT_NAMES[dataset]

        main_figure, _ = build_individual_figure(dataset, MAIN_METHODS)
        combined_figure, _ = build_combined_figure(dataset)
        outputs[dataset] = {
            "main": _save_png(main_figure, folder / names["main"]),
            "combined": _save_png(combined_figure, folder / names["combined"]),
        }
    return outputs

if __name__ == "__main__":
    render_final_pngs(ROOT / "outputs" / "final_results")
