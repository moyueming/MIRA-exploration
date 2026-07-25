from __future__ import annotations

import importlib.util
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from final_figure_registry import GREEDY_EDA, GREEDY_REWARD_FILES

from plot_galaxy_extrinsic_reward_template import (
    ABLATION_METHODS,
    MAIN_METHODS,
    METHOD_STYLES,
    aggregate_seed_curves,
)


ROOT = Path(__file__).resolve().parent
MAX_EPISODE = 1000
SMOOTHING_WINDOWS = {"Galaxy": 25, "Covertype": 50}

GALAXY_LABELS = {
    "MIRA": "MIRA",
    "MIRA-noEXT": "MIRA w/o Ext. Reward",
    "DORA": "DORA",
    "ATENA-ext": "ATENA",
    "ATENA": "ATENA w/o Ext. Reward",
    "A3Cpure": "A3C",
    "Random": "Random",
}

COVERTYPE_LABELS = {
    "MIRA": "MIRA",
    "MIRA (no ext.)": "MIRA w/o Ext. Reward",
    "Pure A3C": "A3C",
    "ATENA-style": "ATENA w/o Ext. Reward",
    "ATENA-style + ext.": "ATENA",
    "DORA": "DORA",
    "Random": "Random",
}


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load configuration module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _dataset_files() -> dict[str, dict[str, tuple[Path, Path, Path]]]:
    galaxy = _load_module("galaxy_final_config", ROOT / "export_galaxy_final_six_png.py")
    covertype = _load_module(
        "covertype_final_config",
        ROOT / "covertype-exploration" / "plot_covertype_final_ma25_figures.py",
    )

    galaxy_files = {
        GALAXY_LABELS[method["label"]]: tuple(Path(path) for path in method["reward_files"])
        for method in galaxy.METHODS
    }
    covertype_files = {
        COVERTYPE_LABELS[label]: tuple(Path(paths[seed]) for seed in (1, 2, 3))
        for label, paths in covertype.REWARD_FILES.items()
    }
    dataset_files = {"Galaxy": galaxy_files, "Covertype": covertype_files}
    for dataset in dataset_files:
        dataset_files[dataset][GREEDY_EDA] = GREEDY_REWARD_FILES[dataset]
    return dataset_files


DATASET_FILES = _dataset_files()


def load_seed_curves(paths: tuple[Path, ...]) -> np.ndarray:
    if len(paths) != 3:
        raise ValueError("exactly three seed paths are required")

    seed_curves = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path)
        required = {"episode", "extrinsic_reward"}
        missing = required.difference(frame.columns)
        if missing:
            raise KeyError(f"{path} is missing columns: {sorted(missing)}")

        frame = frame[["episode", "extrinsic_reward"]].copy()
        frame["episode"] = pd.to_numeric(frame["episode"], errors="coerce")
        frame["extrinsic_reward"] = pd.to_numeric(
            frame["extrinsic_reward"], errors="coerce"
        )
        frame = frame.dropna().sort_values("episode")
        frame["episode"] = frame["episode"].astype(int)
        frame = frame.drop_duplicates("episode", keep="last").set_index("episode")
        values = frame["extrinsic_reward"].reindex(range(1, MAX_EPISODE + 1))
        if values.isna().any():
            missing_episodes = values.index[values.isna()].tolist()
            raise ValueError(f"{path} is missing episodes: {missing_episodes[:10]}")
        seed_curves.append(values.to_numpy(dtype=float))
    return np.vstack(seed_curves)


def _configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 9.5,
            "axes.labelsize": 10.5,
            "axes.titlesize": 11.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 9.2,
            "axes.linewidth": 0.9,
        }
    )


def build_two_dataset_figure(
    methods: tuple[str, ...],
) -> tuple[plt.Figure, np.ndarray]:
    _configure_style()
    episodes = np.arange(1, MAX_EPISODE + 1)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6), squeeze=False)
    axes = axes[0]

    for ax, (dataset, files) in zip(axes, DATASET_FILES.items()):
        lower_bounds = []
        upper_bounds = []
        for method in methods:
            seed_curves = load_seed_curves(files[method])
            mean, sd = aggregate_seed_curves(
                seed_curves,
                window=SMOOTHING_WINDOWS[dataset],
            )
            color, linestyle, linewidth = METHOD_STYLES[method]
            lower = mean - sd
            upper = mean + sd
            lower_bounds.append(np.min(lower))
            upper_bounds.append(np.max(upper))

            ax.fill_between(
                episodes,
                lower,
                upper,
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

        data_min = min(lower_bounds)
        data_max = max(upper_bounds)
        padding = 0.04 * max(data_max - data_min, 1.0)
        ax.set_xlim(0, MAX_EPISODE)
        ax.set_ylim(min(0.0, data_min - padding), data_max + padding)
        ax.set_title(dataset, fontweight="bold", pad=6)
        ax.set_xlabel("Episode", fontweight="bold")
        ax.set_ylabel("Episode Reward", fontweight="bold")
        ax.grid(axis="y", color="#D0D0D0", linewidth=0.5, alpha=0.70)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("black")
            spine.set_linewidth(0.9)
        ax.tick_params(direction="out", length=3.2, width=0.9, colors="black")

    handles, labels = axes[0].get_legend_handles_labels()
    legend = fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=len(methods),
        frameon=False,
        handlelength=2.4,
        columnspacing=1.0,
        handletextpad=0.5,
        prop={"weight": "bold", "size": 9.2},
    )
    for handle in legend.legend_handles:
        handle.set_linewidth(2.8)

    fig.subplots_adjust(top=0.80, bottom=0.19, left=0.09, right=0.965, wspace=0.30)
    return fig, axes


def _save_figure(
    fig: plt.Figure,
    output_dir: Path,
    stem: str,
) -> tuple[Path, Path]:
    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"
    fig.savefig(png_path, dpi=300, facecolor="white")
    fig.savefig(pdf_path, facecolor="white")
    plt.close(fig)
    return png_path, pdf_path


def render_real_figures(output_dir: Path) -> dict[str, tuple[Path, Path]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    main_figure, _ = build_two_dataset_figure(MAIN_METHODS)
    ablation_figure, _ = build_two_dataset_figure(ABLATION_METHODS)
    return {
        "main": _save_figure(
            main_figure,
            output_dir,
            "episode_reward_main_galaxy_covertype",
        ),
        "ablation": _save_figure(
            ablation_figure,
            output_dir,
            "episode_reward_ablation_galaxy_covertype",
        ),
    }


if __name__ == "__main__":
    render_real_figures(ROOT / "outputs" / "final_results" / "episode_reward_v2")

