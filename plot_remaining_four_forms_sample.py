from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from plot_galaxy_extrinsic_reward_template import MAIN_METHODS, METHOD_STYLES


ROOT = Path(__file__).resolve().parent
EPISODES = np.arange(1, 1001)
MARKERS = {
    "MIRA": "o",
    "DORA": "s",
    "ATENA": "D",
    "A3C": "^",
    "Random": "X",
}


def generate_sample_data() -> dict[str, dict[str, np.ndarray]]:
    configs = {
        "MIRA": (5.5, 18.0, 0.95),
        "DORA": (4.0, 10.0, 1.15),
        "ATENA": (2.0, 4.8, 0.70),
        "A3C": (3.0, 7.0, 1.00),
        "Random": (1.4, 2.8, 1.35),
    }
    progress = 0.35 * (1.0 - np.exp(-EPISODES / 230.0))
    progress += 0.65 / (1.0 + np.exp(-(EPISODES - 620.0) / 140.0))
    progress /= progress[-1]

    result: dict[str, dict[str, np.ndarray]] = {}
    for method_index, (method, (start, end, discovery_rate)) in enumerate(
        configs.items()
    ):
        cumulative_rewards = []
        cumulative_sets = []
        last200_efficiency = []
        for seed in range(3):
            rng = np.random.default_rng(20_000 + method_index * 100 + seed)
            reward_mean = start + (end - start) * progress
            reward_scale = 1.0 + rng.normal(0.0, 0.07)
            reward_increment = rng.gamma(
                shape=4.0,
                scale=np.maximum(reward_mean * reward_scale, 0.1) / 4.0,
            )
            rate = discovery_rate * (1.05 - 0.35 * progress)
            set_increment = rng.poisson(np.maximum(rate, 0.05)).astype(float)

            cumulative_rewards.append(np.cumsum(reward_increment))
            cumulative_sets.append(np.cumsum(set_increment))
            episode_efficiency = reward_increment / np.maximum(set_increment, 1.0)
            last200_efficiency.append(float(np.mean(episode_efficiency[-200:])))

        result[method] = {
            "cum_reward": np.vstack(cumulative_rewards),
            "cum_sets": np.vstack(cumulative_sets),
            "last200_eff": np.asarray(last200_efficiency),
        }
    return result


def _configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 10,
            "axes.labelsize": 10.5,
            "axes.titlesize": 11.5,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9.5,
            "axes.linewidth": 0.9,
        }
    )


def _frame(ax: plt.Axes) -> None:
    ax.grid(axis="y", color="#D0D0D0", linewidth=0.5, alpha=0.68)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(0.9)
    ax.tick_params(direction="out", length=3.2, width=0.9, colors="black")


def build_sample_figure() -> tuple[plt.Figure, np.ndarray]:
    _configure_style()
    data = generate_sample_data()
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.4))
    ax_reward, ax_sets, ax_efficiency, ax_pareto = axes.flat

    reward_checkpoints = [199, 399, 599, 799, 999]
    pareto_checkpoints = [249, 499, 749, 999]
    for method in MAIN_METHODS:
        color, _, linewidth = METHOD_STYLES[method]
        marker = MARKERS[method]
        reward_seeds = data[method]["cum_reward"]
        set_seeds = data[method]["cum_sets"]
        reward_mean = reward_seeds.mean(axis=0)
        reward_sd = reward_seeds.std(axis=0, ddof=1)
        set_mean = set_seeds.mean(axis=0)
        set_sd = set_seeds.std(axis=0, ddof=1)

        ax_reward.fill_between(
            EPISODES,
            reward_mean - reward_sd,
            reward_mean + reward_sd,
            color=color,
            alpha=0.12,
            linewidth=0,
        )
        ax_reward.plot(
            EPISODES,
            reward_mean,
            color=color,
            linewidth=linewidth,
            marker=marker,
            markevery=reward_checkpoints,
            markersize=4.2,
            markerfacecolor="white",
            markeredgewidth=0.9,
            label=method,
        )

        ax_sets.fill_between(
            EPISODES,
            set_mean - set_sd,
            set_mean + set_sd,
            step="post",
            color=color,
            alpha=0.12,
            linewidth=0,
        )
        ax_sets.step(
            EPISODES,
            set_mean,
            where="post",
            color=color,
            linewidth=linewidth,
            label=method,
        )

        ax_pareto.plot(
            set_mean,
            reward_mean,
            color=color,
            linewidth=linewidth,
            label=method,
        )
        ax_pareto.scatter(
            set_mean[pareto_checkpoints],
            reward_mean[pareto_checkpoints],
            s=20,
            marker=marker,
            facecolors="white",
            edgecolors=color,
            linewidths=0.9,
            zorder=4,
        )
        ax_pareto.scatter(
            [set_mean[-1]],
            [reward_mean[-1]],
            s=48,
            marker=marker,
            color=color,
            edgecolors="white",
            linewidths=0.7,
            zorder=5,
        )

    ax_reward.set_title("(a) Cumulative reward", fontweight="bold")
    ax_reward.set_xlabel("Episode", fontweight="bold")
    ax_reward.set_ylabel("Cumulative Reward", fontweight="bold")

    ax_sets.set_title("(b) Unique-set discovery", fontweight="bold")
    ax_sets.set_xlabel("Episode", fontweight="bold")
    ax_sets.set_ylabel("Unique Sets Viewed", fontweight="bold")

    method_order = list(MAIN_METHODS)
    y_positions = np.arange(len(method_order))
    for y, method in zip(y_positions, method_order):
        color, _, _ = METHOD_STYLES[method]
        values = data[method]["last200_eff"]
        mean = float(values.mean())
        sd = float(values.std(ddof=1))
        seed_offsets = np.asarray([-0.09, 0.0, 0.09])
        ax_efficiency.scatter(
            values,
            y + seed_offsets,
            s=26,
            color=color,
            alpha=0.72,
            edgecolors="white",
            linewidths=0.5,
            zorder=3,
        )
        ax_efficiency.errorbar(
            mean,
            y,
            xerr=sd,
            fmt="o",
            color=color,
            ecolor=color,
            markersize=6.2,
            markeredgecolor="white",
            markeredgewidth=0.7,
            elinewidth=1.2,
            capsize=3.0,
            zorder=4,
        )
    ax_efficiency.set_yticks(y_positions, method_order, fontweight="bold")
    ax_efficiency.invert_yaxis()
    ax_efficiency.set_title("(c) Last-200 efficiency", fontweight="bold")
    ax_efficiency.set_xlabel("Last-200 Target Efficiency", fontweight="bold")
    ax_efficiency.set_ylabel("Method", fontweight="bold")

    ax_pareto.set_title("(d) Reward-coverage trade-off", fontweight="bold")
    ax_pareto.set_xlabel("Cumulative Unique Sets", fontweight="bold")
    ax_pareto.set_ylabel("Cumulative Reward", fontweight="bold")
    max_x = max(float(data[m]["cum_sets"].mean(axis=0)[-1]) for m in MAIN_METHODS)
    max_y = max(float(data[m]["cum_reward"].mean(axis=0)[-1]) for m in MAIN_METHODS)
    for slope in (3.0, 6.0, 9.0):
        ray_end = min(max_x, max_y / slope)
        ax_pareto.plot(
            [0.0, ray_end],
            [0.0, slope * ray_end],
            color="#888888",
            linestyle="--",
            linewidth=0.65,
            alpha=0.55,
            zorder=0,
        )
        label_x = ray_end * 0.72
        ax_pareto.text(
            label_x,
            slope * label_x,
            f"{slope:g} reward/set",
            color="#666666",
            fontsize=7.5,
            rotation=18,
            ha="left",
            va="bottom",
        )

    for ax in axes.flat:
        _frame(ax)

    handles, labels = ax_reward.get_legend_handles_labels()
    legend = fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=len(MAIN_METHODS),
        frameon=False,
        handlelength=2.6,
        columnspacing=1.2,
        handletextpad=0.5,
        prop={"weight": "bold", "size": 10},
    )
    for handle in legend.legend_handles:
        handle.set_linewidth(2.8)
        handle.set_linestyle("-")

    fig.subplots_adjust(
        top=0.89,
        bottom=0.09,
        left=0.08,
        right=0.975,
        hspace=0.38,
        wspace=0.26,
    )
    return fig, axes


def render_sample(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, _ = build_sample_figure()
    output = output_dir / "remaining_four_figure_forms_sample.png"
    fig.savefig(output, dpi=300, facecolor="white")
    plt.close(fig)
    return output


if __name__ == "__main__":
    render_sample(ROOT / "outputs" / "figure_templates")
