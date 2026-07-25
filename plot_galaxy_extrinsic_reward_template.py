from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from final_figure_registry import ABLATION_METHODS, MAIN_METHODS, METHOD_STYLES


def aggregate_seed_curves(
    seed_curves: np.ndarray,
    window: int = 25,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(seed_curves, dtype=float)
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError(
            "seed_curves must have shape (n_seeds, n_episodes) with n_seeds >= 2"
        )
    if window < 1:
        raise ValueError("window must be positive")

    smoothed = np.vstack(
        [
            pd.Series(seed)
            .rolling(window=window, min_periods=1)
            .mean()
            .to_numpy()
            for seed in values
        ]
    )
    return np.nanmean(smoothed, axis=0), np.nanstd(smoothed, axis=0, ddof=1)


def errorbar_indices(episodes: np.ndarray, every: int = 50) -> np.ndarray:
    values = np.asarray(episodes, dtype=int)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("episodes must be a non-empty one-dimensional array")
    if every < 1:
        raise ValueError("every must be positive")

    mask = (values == values[0]) | (values % every == 0) | (values == values[-1])
    return np.flatnonzero(mask)


def _correlated_noise(
    rng: np.random.Generator,
    size: int,
    scale: float,
    persistence: float = 0.92,
) -> np.ndarray:
    innovations = rng.normal(0.0, scale, size=size)
    noise = np.empty(size, dtype=float)
    noise[0] = innovations[0]
    for index in range(1, size):
        noise[index] = persistence * noise[index - 1] + innovations[index]
    return noise * np.sqrt(1.0 - persistence**2)


def generate_synthetic_curves(episodes: np.ndarray) -> dict[str, np.ndarray]:
    x = np.asarray(episodes, dtype=float)
    progress = 0.42 * (1.0 - np.exp(-x / 230.0))
    progress += 0.58 / (1.0 + np.exp(-(x - 670.0) / 120.0))
    progress /= progress[-1]

    configs = {
        "MIRA": (42.0, 270.0, 30.0, 0.24),
        "MIRA w/o Ext. Reward": (23.0, 38.0, 12.0, 0.13),
        "DORA": (27.0, 61.0, 19.0, 0.25),
        "Greedy": (24.0, 52.0, 15.0, 0.18),
        "ATENA": (7.0, 6.0, 4.0, 0.10),
        "ATENA w/o Ext. Reward": (20.0, 48.0, 17.0, 0.22),
        "A3C": (18.0, 16.0, 9.0, 0.12),
        "Random": (14.0, 18.0, 7.0, 0.10),
    }

    curves: dict[str, np.ndarray] = {}
    for method_index, (name, (start, end, noise, spread)) in enumerate(
        configs.items()
    ):
        seed_curves = []
        base_trend = start + (end - start) * progress
        for seed in range(3):
            rng = np.random.default_rng(10_000 + method_index * 100 + seed)
            seed_scale = 1.0 + rng.normal(0.0, spread)
            phase = rng.uniform(0.0, 2.0 * np.pi)
            oscillation = noise * 0.35 * np.sin(x / 52.0 + phase)
            stochastic = _correlated_noise(rng, x.size, noise)
            curve = base_trend * seed_scale + oscillation + stochastic
            seed_curves.append(np.maximum(0.0, curve))
        curves[name] = np.vstack(seed_curves)
    return curves


def _configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 11,
            "axes.labelsize": 12.5,
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
            "legend.fontsize": 10.2,
            "axes.linewidth": 0.9,
            "lines.solid_capstyle": "round",
        }
    )


def build_figure(
    methods: tuple[str, ...],
    curves: dict[str, np.ndarray],
) -> tuple[plt.Figure, plt.Axes]:
    missing = [method for method in methods if method not in curves]
    if missing:
        raise ValueError(f"missing synthetic curves for: {', '.join(missing)}")

    _configure_style()
    episodes = np.arange(1, 1001)
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    lower_bounds = []
    upper_bounds = []

    for method in methods:
        mean, sd = aggregate_seed_curves(curves[method], window=25)
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
    padding = 0.04 * (data_max - data_min)
    ax.set_xlim(0, 1000)
    ax.set_ylim(min(0.0, data_min - padding), data_max + padding)
    ax.set_xlabel("Episode", fontweight="bold")
    ax.set_ylabel("Episode Reward", fontweight="bold")
    ax.grid(axis="y", color="#D0D0D0", linewidth=0.55, alpha=0.72)
    ax.set_axisbelow(True)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(0.9)

    ax.tick_params(direction="out", length=3.5, width=0.9, colors="black")
    legend = ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.015),
        ncol=len(methods),
        frameon=False,
        handlelength=2.6,
        columnspacing=1.15,
        handletextpad=0.55,
        prop={"weight": "bold", "size": 10.2},
    )
    for handle in legend.legend_handles:
        handle.set_linewidth(2.8)

    fig.subplots_adjust(top=0.79, bottom=0.20, left=0.12, right=0.96)
    return fig, ax


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


def render_templates(output_dir: Path) -> dict[str, tuple[Path, Path]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    curves = generate_synthetic_curves(np.arange(1, 1001))

    main_figure, _ = build_figure(MAIN_METHODS, curves)
    ablation_figure, _ = build_figure(ABLATION_METHODS, curves)
    return {
        "main": _save_figure(
            main_figure,
            output_dir,
            "galaxy_episode_reward_main_template",
        ),
        "ablation": _save_figure(
            ablation_figure,
            output_dir,
            "galaxy_episode_reward_ablation_template",
        ),
    }


if __name__ == "__main__":
    render_templates(Path("outputs") / "figure_templates")

