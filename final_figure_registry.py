from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parent
GREEDY_EDA = "Greedy"

MAIN_METHODS = (
    "MIRA",
    "DORA",
    GREEDY_EDA,
    "ATENA",
    "A3C",
    "Random",
)

ABLATION_METHODS = (
    "MIRA",
    "MIRA w/o Ext. Reward",
    "ATENA",
    "ATENA w/o Ext. Reward",
)

ALL_METHODS = (
    "MIRA",
    "MIRA w/o Ext. Reward",
    "DORA",
    GREEDY_EDA,
    "ATENA",
    "ATENA w/o Ext. Reward",
    "A3C",
    "Random",
)

METHOD_STYLES = {
    "MIRA": ("#0072B2", "-", 1.30),
    "MIRA w/o Ext. Reward": ("#56B4E9", "-", 1.15),
    "DORA": ("#D55E00", "-", 1.15),
    GREEDY_EDA: ("#7B3294", "-", 1.15),
    "ATENA": ("#CC79A7", "-", 1.15),
    "ATENA w/o Ext. Reward": ("#E69F00", "-", 1.15),
    "A3C": ("#009E73", "-", 1.15),
    "Random": ("#666666", "-", 1.15),
}

METHOD_MARKERS = {
    "MIRA": "o",
    "MIRA w/o Ext. Reward": "v",
    "DORA": "s",
    GREEDY_EDA: "H",
    "ATENA": "D",
    "ATENA w/o Ext. Reward": "P",
    "A3C": "^",
    "Random": "X",
}

DISPLAY_LABELS = {
    "MIRA": "MIRA",
    "MIRA w/o Ext. Reward": "MIRA(w/o Ext)",
    "DORA": "DORA",
    GREEDY_EDA: GREEDY_EDA,
    "ATENA": "ATENA",
    "ATENA w/o Ext. Reward": "ATENA(w/o Ext)",
    "A3C": "A3C",
    "Random": "Random",
}

GREEDY_REWARD_FILES = {
    "Galaxy": tuple(
        ROOT
        / "outputs"
        / "GreedyEDA"
        / f"greedy_eda_seed{seed}_greedy_eda_rewards.csv"
        for seed in (1, 2, 3)
    ),
    "Covertype": tuple(
        ROOT
        / "covertype-exploration"
        / "outputs"
        / "GreedyEDA"
        / f"greedy_eda_seed{seed}_full_greedy_eda_rewards.csv"
        for seed in (1, 2, 3)
    ),
}

GREEDY_TRACE_FILES = {
    "Galaxy": tuple(
        ROOT
        / "outputs"
        / "GreedyEDA"
        / f"greedy_eda_seed{seed}_greedy_eda_exploration_trace.csv"
        for seed in (1, 2, 3)
    ),
    "Covertype": tuple(
        ROOT
        / "covertype-exploration"
        / "outputs"
        / "GreedyEDA"
        / f"greedy_eda_seed{seed}_full_greedy_eda_exploration_trace.csv"
        for seed in (1, 2, 3)
    ),
}
