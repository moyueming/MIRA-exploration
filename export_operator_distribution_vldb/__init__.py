from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from final_figure_registry import ALL_METHODS, GREEDY_EDA, GREEDY_TRACE_FILES


OPERATOR_KEYS = ("facet", "superset", "neighbor", "distribution")
OPERATOR_LABELS = ("by_facet", "by_superset", "by_neighbors", "by_distribution")
OPERATOR_COLORS = ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728")
ROOT = Path(__file__).resolve().parent.parent
SEEDS = (1, 2, 3)
METHOD_DISPLAY_ORDER = ALL_METHODS

def _order_methods(methods: list[dict[str, object]]) -> list[dict[str, object]]:
    positions = {label: index for index, label in enumerate(METHOD_DISPLAY_ORDER)}
    return sorted(methods, key=lambda method: positions[str(method["display_label"])])



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


def operator_ymax(means: np.ndarray, deviations: np.ndarray) -> float:
    upper_errorbar = float(np.nanmax(means + deviations))
    padded = max(1.05, upper_errorbar + max(0.04, upper_errorbar * 0.05))
    return float(np.ceil(padded / 0.05) * 0.05)


def format_method_tick_label(label: str) -> str:
    if label == "MIRA w/o Ext. Reward":
        return "MIRA w/o\nExt. Reward"
    if label == "ATENA w/o Ext. Reward":
        return "ATENA w/o\nExt. Reward"
    return label


def plot_operator_distribution(
    methods: list[dict[str, object]],
    output_stem: Path,
    *,
    write_pdf: bool = True,
) -> tuple[Path, Path | None]:
    means: list[np.ndarray] = []
    deviations: list[np.ndarray] = []
    for method in methods:
        matrix = operator_ratio_matrix(method["trace_files"])
        means.append(matrix.mean(axis=0))
        deviations.append(
            matrix.std(axis=0, ddof=1) if matrix.shape[0] > 1 else np.zeros(4)
        )

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(methods), dtype=float)
    width = 0.18
    offsets = (np.arange(4, dtype=float) - 1.5) * width

    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 9,
            "axes.labelsize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.linewidth": 1.0,
            "savefig.dpi": 300,
        }
    )
    fig, ax = plt.subplots(figsize=(7.2, 3.45))
    mean_array = np.vstack(means)
    sd_array = np.vstack(deviations)
    for index, (label, color) in enumerate(
        zip(OPERATOR_LABELS, OPERATOR_COLORS)
    ):
        ax.bar(
            x + offsets[index],
            mean_array[:, index],
            width=width,
            yerr=sd_array[:, index],
            label=label,
            color=color,
            edgecolor="white",
            linewidth=0.45,
            error_kw={"ecolor": "black", "elinewidth": 0.8, "capsize": 2},
            zorder=3,
        )

    ax.set_ylabel("Selection ratio")
    ax.set_xticks(x)
    ax.set_xticklabels([format_method_tick_label(str(method.get("display_label", method.get("label")))) for method in methods])
    ax.set_ylim(0.0, operator_ymax(mean_array, sd_array))
    ax.set_yticks(np.linspace(0.0, 1.0, 6))
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.6, alpha=0.7, zorder=0)
    ax.legend(loc="upper left", frameon=True, fancybox=False, ncol=1)
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    ax.tick_params(direction="out", length=3.5, width=0.9)
    fig.tight_layout(pad=0.6)

    png_path = output_stem.with_suffix(".png")
    fig.savefig(png_path, bbox_inches="tight", pad_inches=0.03)
    pdf_path = output_stem.with_suffix(".pdf") if write_pdf else None
    if pdf_path is not None:
        fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    return png_path, pdf_path


def galaxy_methods() -> list[dict[str, object]]:
    outputs = ROOT / "outputs"
    methods = [
        {
            "display_label": "MIRA",
            "trace_files": [
                outputs
                / "our MIRA"
                / f"galaxy_bile_fixed_seed{seed}_final"
                / f"galaxy_bile_fixed_seed{seed}_final_exploration_trace.csv"
                for seed in SEEDS
            ],
        },
        {
            "display_label": "MIRA w/o Ext. Reward",
            "trace_files": [
                outputs
                / "our MIRA-no extrinsic"
                / f"galaxy_bile_no_ext_fixed_seed{seed}_final"
                / f"galaxy_bile_no_ext_fixed_seed{seed}_final_exploration_trace.csv"
                for seed in SEEDS
            ],
        },
        {
            "display_label": "A3C",
            "trace_files": [
                outputs
                / "A3Cpure"
                / f"galaxy_pure_a3c_w5_seed{seed}"
                / f"galaxy_pure_a3c_w5_seed{seed}_pure_a3c_exploration_trace.csv"
                for seed in SEEDS
            ],
        },
        {
            "display_label": "ATENA w/o Ext. Reward",
            "trace_files": [
                outputs / "ATENA_pure" / "atena_pure_seed_1_exploration_trace.csv",
                outputs / "ATENA_pure" / "atena_pure_seed_2_exploration_trace.csv",
                outputs
                / "ATENA_pure"
                / "galaxy_ATENA_pure_fixed_seed3_final_exploration_trace.csv",
            ],
        },
        {
            "display_label": "ATENA",
            "trace_files": [
                outputs / "ATENA_ext" / "atena_ext_seed_1_exploration_trace.csv",
                None,
                outputs
                / "ATENA_ext"
                / "galaxy_ATENA_ext_fixed_seed3_final_exploration_trace.csv",
            ],
        },
        {
            "display_label": "DORA",
            "trace_files": [
                outputs / "DORA" / "paper_a3c_seed_1_exploration_trace.csv",
                outputs / "DORA" / "paper_a3c_seed_2_exploration_trace.csv",
                outputs
                / "DORA"
                / "galaxy_paper_a3c_seed3_exploration_trace.csv",
            ],
        },
        {
            "display_label": GREEDY_EDA,
            "trace_files": list(GREEDY_TRACE_FILES["Galaxy"]),
        },
        {
            "display_label": "Random",
            "trace_files": [
                outputs
                / "random"
                / f"baseline_random_seed_{seed}_precomputed_random_exploration_trace.csv"
                for seed in SEEDS
            ],
        },
    ]
    return _order_methods(methods)


def covertype_methods() -> list[dict[str, object]]:
    outputs = ROOT / "covertype-exploration" / "outputs"
    methods = [
        {
            "display_label": "MIRA",
            "trace_files": [
                outputs
                / "MIRA"
                / f"mira_seed{seed}_v6"
                / f"mira_seed{seed}_v6_mira_exploration_trace.csv"
                for seed in SEEDS
            ],
        },
        {
            "display_label": "MIRA w/o Ext. Reward",
            "trace_files": [
                outputs
                / "MIRA_noEXT"
                / f"mira_no_ext_seed{seed}_v6"
                / f"mira_no_ext_seed{seed}_v6_mira_no_ext_exploration_trace.csv"
                for seed in SEEDS
            ],
        },
        {
            "display_label": "A3C",
            "trace_files": [
                outputs
                / "A3Cpure"
                / f"pure_a3c_seed{seed}_full"
                / f"pure_a3c_seed{seed}_full_pure_a3c_exploration_trace.csv"
                for seed in SEEDS
            ],
        },
        {
            "display_label": "ATENA w/o Ext. Reward",
            "trace_files": [
                outputs
                / "ATENA"
                / f"atena_seed{seed}_full"
                / f"atena_seed{seed}_full_atena_exploration_trace.csv"
                for seed in SEEDS
            ],
        },
        {
            "display_label": "ATENA",
            "trace_files": [
                outputs
                / "ATENA-EXT"
                / f"atena_ext_seed{seed}_full"
                / f"atena_ext_seed{seed}_full_atena_extrinsic_exploration_trace.csv"
                for seed in SEEDS
            ],
        },
        {
            "display_label": "DORA",
            "trace_files": [
                outputs
                / "DORA"
                / f"paper_a3c_seed{seed}_full"
                / f"paper_a3c_seed{seed}_full_paper_a3c_exploration_trace.csv"
                for seed in SEEDS
            ],
        },
        {
            "display_label": GREEDY_EDA,
            "trace_files": list(GREEDY_TRACE_FILES["Covertype"]),
        },
        {
            "display_label": "Random",
            "trace_files": [
                outputs
                / "random"
                / f"random_seed{seed}_random_exploration_trace.csv"
                for seed in SEEDS
            ],
        },
    ]
    return _order_methods(methods)


def main() -> None:
    plot_operator_distribution(
        galaxy_methods(),
        ROOT
        / "outputs"
        / "final_results"
        / "galaxy"
        / "galaxy_operator_distribution_vldb",
    )
    plot_operator_distribution(
        covertype_methods(),
        ROOT
        / "outputs"
        / "final_results"
        / "covertype"
        / "covertype_operator_distribution_vldb",
    )
