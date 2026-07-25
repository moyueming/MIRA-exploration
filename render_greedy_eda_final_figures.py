from __future__ import annotations

import hashlib
from pathlib import Path

from export_operator_distribution_vldb import (
    covertype_methods,
    galaxy_methods,
    plot_operator_distribution,
)
from plot_final_compact_2x4 import render_final_compact_pngs
from plot_final_cumulative_performance import (
    render_main_containing_cumulative_pngs,
)
from plot_final_episode_reward_by_dataset import (
    render_main_containing_episode_pngs,
)


ROOT = Path(__file__).resolve().parent

FINAL_FOLDERS = {
    "Galaxy": "galaxy_final",
    "Covertype": "covertype_final",
}

TARGET_OUTPUTS = {
    dataset: (
        f"{prefix}_cumulative_reward_main.png",
        f"{prefix}_cumulative_target_efficiency_main.png",
        f"{prefix}_cumulative_unique_sets_main.png",
        f"{prefix}_cumulative_reward_all_methods.png",
        f"{prefix}_cumulative_target_efficiency_all_methods.png",
        f"{prefix}_cumulative_unique_sets_all_methods.png",
        f"{prefix}_episode_reward_main.png",
        f"{prefix}_episode_reward_combined.png",
        f"{prefix}_compact_2x4.png",
        f"{prefix}_operator_distribution_vldb.png",
    )
    for dataset, prefix in (("Galaxy", "galaxy"), ("Covertype", "covertype"))
}

EXCLUDED_ABLATION_OUTPUTS = {
    dataset: (
        f"{prefix}_cumulative_reward_ablation.png",
        f"{prefix}_cumulative_target_efficiency_ablation.png",
        f"{prefix}_cumulative_unique_sets_ablation.png",
        f"{prefix}_episode_reward_ablation.png",
    )
    for dataset, prefix in (("Galaxy", "galaxy"), ("Covertype", "covertype"))
}


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _protected_digests(final_root: Path) -> dict[Path, str]:
    return {
        path: _digest(path)
        for dataset, folder_name in FINAL_FOLDERS.items()
        for name in EXCLUDED_ABLATION_OUTPUTS[dataset]
        if (path := final_root / folder_name / name).exists()
    }


def _verify_protected_files(before: dict[Path, str]) -> None:
    changed = [path for path, expected in before.items() if _digest(path) != expected]
    if changed:
        raise RuntimeError(f"Protected ablation outputs changed: {changed}")


def render_greedy_eda_final_figures(
    final_root: Path,
) -> dict[str, dict[str, Path]]:
    before = _protected_digests(final_root)
    cumulative = render_main_containing_cumulative_pngs(final_root)
    episode = render_main_containing_episode_pngs(final_root)
    compact = render_final_compact_pngs(final_root)

    outputs: dict[str, dict[str, Path]] = {
        "Galaxy": {},
        "Covertype": {},
    }
    for dataset in outputs:
        for path in cumulative[dataset].values():
            outputs[dataset][path.name] = path
        for path in episode[dataset].values():
            outputs[dataset][path.name] = path
        compact_path = compact[dataset]
        outputs[dataset][compact_path.name] = compact_path

    for dataset, methods_factory in (
        ("Galaxy", galaxy_methods),
        ("Covertype", covertype_methods),
    ):
        prefix = dataset.lower()
        stem = (
            final_root
            / FINAL_FOLDERS[dataset]
            / f"{prefix}_operator_distribution_vldb"
        )
        png_path, pdf_path = plot_operator_distribution(
            methods_factory(),
            stem,
            write_pdf=False,
        )
        if pdf_path is not None:
            raise RuntimeError(f"Unexpected PDF output: {pdf_path}")
        outputs[dataset][png_path.name] = png_path

    _verify_protected_files(before)
    for dataset, folder_name in FINAL_FOLDERS.items():
        expected = set(TARGET_OUTPUTS[dataset])
        if set(outputs[dataset]) != expected:
            raise RuntimeError(
                f"Unexpected {dataset} target set: {sorted(outputs[dataset])}"
            )
        pdfs = sorted((final_root / folder_name).glob("*.pdf"))
        if pdfs:
            raise RuntimeError(f"PDF files found in {folder_name}: {pdfs}")

    return outputs


if __name__ == "__main__":
    render_greedy_eda_final_figures(ROOT / "outputs" / "final_results")
