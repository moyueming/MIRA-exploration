from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
FINAL_DIR = OUTPUTS / "final_results"
MAX_EPISODE = 1000


METHODS = [
    {
        "label": "MIRA",
        "color": "#1f77b4",
        "reward_files": [
            OUTPUTS / "our MIRA" / f"galaxy_bile_fixed_seed{seed}_final" / f"galaxy_bile_fixed_seed{seed}_final_fusion_rewards.csv"
            for seed in (1, 2, 3)
        ],
        "trace_files": [
            OUTPUTS / "our MIRA" / f"galaxy_bile_fixed_seed{seed}_final" / f"galaxy_bile_fixed_seed{seed}_final_exploration_trace.csv"
            for seed in (1, 2, 3)
        ],
    },
    {
        "label": "MIRA-noEXT",
        "color": "#ff7f0e",
        "reward_files": [
            OUTPUTS / "our MIRA-no extrinsic" / f"galaxy_bile_no_ext_fixed_seed{seed}_final" / f"galaxy_bile_no_ext_fixed_seed{seed}_final_fusion_rewards.csv"
            for seed in (1, 2, 3)
        ],
        "trace_files": [
            OUTPUTS / "our MIRA-no extrinsic" / f"galaxy_bile_no_ext_fixed_seed{seed}_final" / f"galaxy_bile_no_ext_fixed_seed{seed}_final_exploration_trace.csv"
            for seed in (1, 2, 3)
        ],
    },
    {
        "label": "DORA",
        "color": "#2ca02c",
        "reward_files": [
            OUTPUTS / "DORA" / "paper_a3c_seed_1_paper_a3c_rewards.csv",
            OUTPUTS / "DORA" / "paper_a3c_seed_2_paper_a3c_rewards.csv",
            OUTPUTS / "DORA" / "galaxy_paper_a3c_seed3_paper_a3c_rewards.csv",
        ],
        "trace_files": [
            OUTPUTS / "DORA" / "paper_a3c_seed_1_exploration_trace.csv",
            OUTPUTS / "DORA" / "paper_a3c_seed_2_exploration_trace.csv",
            OUTPUTS / "DORA" / "galaxy_paper_a3c_seed3_exploration_trace.csv",
        ],
    },
    {
        "label": "ATENA-ext",
        "color": "#d62728",
        "reward_files": [
            OUTPUTS / "ATENA_ext" / "atena_ext_seed_1_fusion_rewards.csv",
            OUTPUTS / "ATENA_ext" / "galaxy_ATENA_ext_fixed_seed2_final_fusion_rewards.csv",
            OUTPUTS / "ATENA_ext" / "galaxy_ATENA_ext_fixed_seed3_final_fusion_rewards.csv",
        ],
        "trace_files": [
            OUTPUTS / "ATENA_ext" / "atena_ext_seed_1_exploration_trace.csv",
            None,
            OUTPUTS / "ATENA_ext" / "galaxy_ATENA_ext_fixed_seed3_final_exploration_trace.csv",
        ],
    },
    {
        "label": "ATENA",
        "color": "#9467bd",
        "reward_files": [
            OUTPUTS / "ATENA_pure" / "atena_pure_seed_1_fusion_rewards.csv",
            OUTPUTS / "ATENA_pure" / "atena_pure_seed_2_fusion_rewards.csv",
            OUTPUTS / "ATENA_pure" / "galaxy_ATENA_pure_fixed_seed3_final_fusion_rewards.csv",
        ],
        "trace_files": [
            OUTPUTS / "ATENA_pure" / "atena_pure_seed_1_exploration_trace.csv",
            OUTPUTS / "ATENA_pure" / "atena_pure_seed_2_exploration_trace.csv",
            OUTPUTS / "ATENA_pure" / "galaxy_ATENA_pure_fixed_seed3_final_exploration_trace.csv",
        ],
    },
    {
        "label": "A3Cpure",
        "color": "#8c564b",
        "reward_files": [
            OUTPUTS / "A3Cpure" / f"galaxy_pure_a3c_w5_seed{seed}" / f"galaxy_pure_a3c_w5_seed{seed}_pure_a3c_rewards.csv"
            for seed in (1, 2, 3)
        ],
        "trace_files": [
            OUTPUTS / "A3Cpure" / f"galaxy_pure_a3c_w5_seed{seed}" / f"galaxy_pure_a3c_w5_seed{seed}_pure_a3c_exploration_trace.csv"
            for seed in (1, 2, 3)
        ],
    },
    {
        "label": "Random",
        "color": "#7f7f7f",
        "reward_files": [
            OUTPUTS / "random" / f"baseline_random_seed_{seed}_precomputed_random_rewards.csv"
            for seed in (1, 2, 3)
        ],
        "trace_files": [
            OUTPUTS / "random" / f"baseline_random_seed_{seed}_precomputed_random_exploration_trace.csv"
            for seed in (1, 2, 3)
        ],
    },
]


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 23,
            "axes.titlesize": 30,
            "axes.labelsize": 26,
            "xtick.labelsize": 22,
            "ytick.labelsize": 22,
            "legend.fontsize": 20,
            "figure.titlesize": 30,
            "axes.linewidth": 1.8,
            "lines.linewidth": 3.2,
            "savefig.dpi": 300,
            "figure.dpi": 140,
        }
    )


def read_reward_file(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "episode" not in df.columns:
        df.insert(0, "episode", np.arange(1, len(df) + 1))
    df = df.sort_values("episode").drop_duplicates("episode", keep="last")
    return df[df["episode"].between(1, MAX_EPISODE)].copy()


def mean_metric(files: list[Path], metric: str) -> tuple[np.ndarray, np.ndarray]:
    series = []
    for path in files:
        if not path.exists():
            continue
        df = read_reward_file(path)
        if metric not in df.columns:
            continue
        values = pd.to_numeric(df[metric], errors="coerce")
        item = pd.Series(values.to_numpy(dtype=float), index=df["episode"].astype(int))
        series.append(item)
    if not series:
        return np.array([], dtype=int), np.array([], dtype=float)
    aligned = pd.concat(series, axis=1).sort_index()
    aligned = aligned.reindex(range(1, min(MAX_EPISODE, int(aligned.index.max())) + 1))
    return aligned.index.to_numpy(dtype=int), aligned.mean(axis=1, skipna=True).to_numpy(dtype=float)


def plot_metric(metric: str, ylabel: str, title: str, filename: str) -> None:
    fig, ax = plt.subplots(figsize=(13.5, 8.4))
    for method in METHODS:
        x, y = mean_metric(method["reward_files"], metric)
        if x.size == 0:
            continue
        ax.plot(x, y, label=method["label"], color=method["color"])
    ax.set_title(title, fontweight="bold", pad=16)
    ax.set_xlabel("Episode")
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", linewidth=1.0, alpha=0.32)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="best", frameon=True, fancybox=True, framealpha=0.92)
    fig.tight_layout()
    fig.savefig(FINAL_DIR / filename, bbox_inches="tight")
    plt.close(fig)


def normalize_operator(name: object) -> str:
    text = str(name)
    lower = text.lower()
    if "facet" in lower:
        return "Facet"
    if "neighbor" in lower:
        return "Neighbor"
    if "superset" in lower:
        return "Superset"
    if "distribution" in lower:
        return "Distribution"
    return "Other"


def trace_counts(path: Path, last_episodes: int | None = None) -> pd.Series:
    if path is None or not path.exists():
        return pd.Series(dtype=float)
    usecols = ["episode", "operator"]
    df = pd.read_csv(path, usecols=lambda col: col in usecols)
    if "operator" not in df.columns:
        return pd.Series(dtype=float)
    if "episode" in df.columns and last_episodes is not None and not df.empty:
        max_episode = int(pd.to_numeric(df["episode"], errors="coerce").max())
        df = df[pd.to_numeric(df["episode"], errors="coerce") > max_episode - int(last_episodes)]
    ops = df["operator"].map(normalize_operator)
    return ops.value_counts()


def mean_operator_distribution(trace_files: list[Path | None], last_episodes: int | None = None) -> pd.Series:
    rows = []
    for path in trace_files:
        counts = trace_counts(path, last_episodes=last_episodes)
        if counts.sum() > 0:
            rows.append(counts / counts.sum())
    if not rows:
        return pd.Series(dtype=float)
    return pd.concat(rows, axis=1).fillna(0.0).mean(axis=1)


def plot_operator_distribution(last_episodes: int | None, title: str, filename: str) -> None:
    categories = ["Facet", "Neighbor", "Superset", "Distribution", "Other"]
    method_data = []
    for method in METHODS:
        dist = mean_operator_distribution(method["trace_files"], last_episodes=last_episodes)
        method_data.append([float(dist.get(category, 0.0)) for category in categories])

    x = np.arange(len(categories))
    width = 0.11
    fig, ax = plt.subplots(figsize=(15.5, 8.7))
    for idx, method in enumerate(METHODS):
        offset = (idx - (len(METHODS) - 1) / 2.0) * width
        ax.bar(
            x + offset,
            method_data[idx],
            width=width,
            label=method["label"],
            color=method["color"],
            edgecolor="white",
            linewidth=0.8,
        )
    ax.set_title(title, fontweight="bold", pad=16)
    ax.set_xlabel("Operator Family")
    ax.set_ylabel("Selection Ratio")
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylim(0.0, max(0.55, min(1.0, math.ceil(max(max(row) for row in method_data) * 10) / 10 + 0.1)))
    ax.grid(True, axis="y", linestyle="--", linewidth=1.0, alpha=0.32)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.02), frameon=True, fontsize=18)
    fig.tight_layout()
    fig.savefig(FINAL_DIR / filename, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    configure_style()
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    plot_metric(
        "extrinsic_reward",
        "Extrinsic Reward",
        "Galaxy Extrinsic Reward",
        "galaxy_extrinsic_reward.png",
    )
    plot_metric(
        "cumulative_extrinsic_reward",
        "Cumulative Extrinsic Reward",
        "Galaxy Cumulative Extrinsic Reward",
        "galaxy_cumulative_extrinsic_reward.png",
    )
    plot_metric(
        "cumulative_unique_sets_viewed",
        "Cumulative Unique Sets",
        "Galaxy Cumulative Unique Sets Viewed",
        "galaxy_cumulative_unique_sets_viewed.png",
    )
    plot_metric(
        "cumulative_target_efficiency",
        "Cumulative Target Efficiency",
        "Galaxy Cumulative Target Efficiency",
        "galaxy_cumulative_target_efficiency.png",
    )
    plot_operator_distribution(
        None,
        "Galaxy Operator Distribution",
        "galaxy_operator_distribution_evolution.png",
    )
    plot_operator_distribution(
        200,
        "Galaxy Operator Distribution in Last 200 Episodes",
        "galaxy_operator_distribution_last200.png",
    )


if __name__ == "__main__":
    main()
