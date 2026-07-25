import argparse
import csv
import json
import re
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


METRICS = ["Precision", "T-BLEU-1", "T-BLEU-2", "T-BLEU-3", "EDA-Sim"]
METHOD_ORDER = ["official_atena", "random", "pure_a3c", "dora", "greedy", "MIRA"]
DISPLAY_NAMES = {
    "official_atena": "Official ATENA",
    "random": "Random",
    "pure_a3c": "Pure A3C",
    "dora": "DORA",
    "greedy": "Greedy",
    "MIRA": "MIRA",
}
EXPECTED_KEYS = {(schema, dataset) for schema in ("flights", "cyber") for dataset in range(1, 5)}
RESERVED_BASELINE_METHODS = {"official_atena", "random", "pure_a3c", "dora", "greedy"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build the clean final ATENA comparison CSVs and table artifacts."
    )
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--output_dir", default="results/final_atena")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--official_summary", default=None)
    parser.add_argument("--random_dir", default=None)
    parser.add_argument("--pure_a3c_dir", default=None)
    parser.add_argument("--dora_dir", default=None)
    parser.add_argument("--greedy_dir", default=None)
    parser.add_argument("--mira_dir", default=None)
    parser.add_argument("--mira_method", default="MIRA")
    parser.add_argument("--mira_display_name", default="MIRA")
    parser.add_argument("--mira_row_source", choices=["last", "final_json"], default="last")
    parser.add_argument("--package_name", default="final_atena_results_package.zip")
    args = parser.parse_args()
    if args.mira_method in RESERVED_BASELINE_METHODS:
        parser.error(f"--mira_method cannot use reserved baseline name: {args.mira_method}")
    return args


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    global METHOD_ORDER, DISPLAY_NAMES
    METHOD_ORDER = ["official_atena", "random", "pure_a3c", "dora", "greedy", args.mira_method]
    DISPLAY_NAMES = {
        "official_atena": "Official ATENA",
        "random": "Random",
        "pure_a3c": "Pure A3C",
        "dora": "DORA",
        "greedy": "Greedy",
        args.mira_method: args.mira_display_name,
    }

    paths = {
        "official_summary": Path(args.official_summary) if args.official_summary else results_dir / "official_atena_summary.csv",
        "random_dir": Path(args.random_dir) if args.random_dir else results_dir / "random",
        "pure_a3c_dir": Path(args.pure_a3c_dir) if args.pure_a3c_dir else results_dir / "pure_a3c",
        "dora_dir": Path(args.dora_dir) if args.dora_dir else results_dir / "dora",
        "greedy_dir": Path(args.greedy_dir) if args.greedy_dir else results_dir / "greedy",
        "mira_dir": Path(args.mira_dir) if args.mira_dir else results_dir / args.mira_method,
    }

    detail_rows = []
    detail_rows.extend(collect_official(paths["official_summary"], args.seed))
    detail_rows.extend(collect_method_dir(paths["random_dir"], "random", args.seed))
    detail_rows.extend(collect_method_dir(paths["pure_a3c_dir"], "pure_a3c", args.seed))
    detail_rows.extend(collect_method_dir(paths["dora_dir"], "dora", args.seed))
    detail_rows.extend(collect_method_dir(paths["greedy_dir"], "greedy", args.seed))
    detail_rows.extend(collect_method_dir(paths["mira_dir"], args.mira_method, args.seed, args.mira_row_source))

    detail = pd.DataFrame(detail_rows)
    validate_detail(detail)
    detail = order_detail(detail)

    detail_path = output_dir / "final_atena_detail_6methods_8datasets.csv"
    all_in_one_path = output_dir / "final_atena_results_all_in_one.csv"
    average_path = output_dir / "final_atena_average_table.csv"
    by_schema_path = output_dir / "final_atena_by_schema.csv"
    markdown_path = output_dir / "final_atena_table.md"
    latex_path = output_dir / "final_atena_table.tex"
    png_path = output_dir / "final_atena_table.png"
    package_path = output_dir / args.package_name

    detail.to_csv(detail_path, index=False)

    average = build_average(detail)
    by_schema = build_by_schema(detail)
    all_in_one = build_all_in_one(detail, by_schema, average)

    average.to_csv(average_path, index=False)
    by_schema.to_csv(by_schema_path, index=False)
    all_in_one.to_csv(all_in_one_path, index=False)

    table = format_table(average)
    markdown_path.write_text(table.to_markdown(index=False), encoding="utf-8")
    latex_path.write_text(table.to_latex(index=False, escape=False), encoding="utf-8")
    render_table_png(table, png_path)

    write_package(package_path, [
        all_in_one_path,
        detail_path,
        average_path,
        by_schema_path,
        markdown_path,
        latex_path,
        png_path,
    ])

    print("Final ATENA artifacts written:")
    for path in [
        all_in_one_path,
        detail_path,
        average_path,
        by_schema_path,
        markdown_path,
        latex_path,
        png_path,
        package_path,
    ]:
        print(path)


def collect_official(summary_path: Path, seed: int):
    if not summary_path.exists():
        raise SystemExit(f"Missing official ATENA summary: {summary_path}")

    df = pd.read_csv(summary_path)
    if "method" in df.columns:
        df = df[df["method"] == "official_atena"]
    if "seed" in df.columns:
        df = df[df["seed"].astype(int) == int(seed)]

    rows = []
    for _, row in df.iterrows():
        normalized = normalize_row(row.to_dict(), "official_atena")
        normalized["source_path"] = str(row.get("load", summary_path))
        normalized["steps"] = normalized.get("steps") or parse_steps_from_path(normalized["source_path"])
        rows.append(normalized)
    return rows


def collect_method_dir(method_dir: Path, method: str, seed: int, row_source: str = "last"):
    if not method_dir.exists():
        raise SystemExit(f"Missing {method} directory: {method_dir}")

    rows = []
    for seed_dir in sorted(method_dir.glob(f"*/seed{int(seed)}")):
        row, source_path = read_run_row(seed_dir, row_source)
        if not row:
            continue
        normalized = normalize_row(row, method)
        normalized["source_path"] = str(source_path)
        rows.append(normalized)
    return rows


def read_run_row(seed_dir: Path, row_source: str):
    if row_source == "final_json":
        final_path = seed_dir / "final_metrics.json"
        if final_path.exists():
            return json.loads(final_path.read_text(encoding="utf-8")), final_path

    metrics_path = seed_dir / "eval_metrics.csv"
    if not metrics_path.exists():
        return None, metrics_path
    file_rows = read_csv_rows(metrics_path)
    return (file_rows[-1], metrics_path) if file_rows else (None, metrics_path)


def normalize_row(row, method: str):
    schema = str(row["schema"])
    dataset = int(row["dataset"])
    seed = int(row.get("seed", 0))
    normalized = {
        "method": method,
        "display_method": DISPLAY_NAMES[method],
        "schema": schema,
        "dataset": dataset,
        "seed": seed,
        "steps": safe_int(row.get("steps", "")),
        "episode_reward": safe_float(row.get("episode_reward", "")),
        "source_path": "",
    }
    for metric in METRICS:
        normalized[metric] = float(row[metric])
    return normalized


def read_csv_rows(path: Path):
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def validate_detail(detail: pd.DataFrame):
    if detail.empty:
        raise SystemExit("No result rows were collected.")

    errors = []
    for method in METHOD_ORDER:
        subset = detail[detail["method"] == method]
        keys = {(str(row.schema), int(row.dataset)) for row in subset.itertuples()}
        missing = sorted(EXPECTED_KEYS - keys)
        extra = sorted(keys - EXPECTED_KEYS)
        if len(subset) != 8 or missing or extra:
            errors.append(
                f"{method}: rows={len(subset)}, missing={missing or 'none'}, extra={extra or 'none'}"
            )

    duplicate_cols = ["method", "schema", "dataset", "seed"]
    duplicates = detail[detail.duplicated(duplicate_cols, keep=False)]
    if not duplicates.empty:
        errors.append("Duplicate method/schema/dataset/seed rows detected.")

    if errors:
        raise SystemExit("Final result validation failed:\n" + "\n".join(errors))


def order_detail(detail: pd.DataFrame):
    method_rank = {method: idx for idx, method in enumerate(METHOD_ORDER)}
    schema_rank = {"flights": 0, "cyber": 1}
    detail = detail.copy()
    detail["_method_rank"] = detail["method"].map(method_rank)
    detail["_schema_rank"] = detail["schema"].map(schema_rank)
    detail = detail.sort_values(["_method_rank", "_schema_rank", "dataset"])
    return detail.drop(columns=["_method_rank", "_schema_rank"])


def build_average(detail: pd.DataFrame):
    average = detail.groupby("method", sort=False)[METRICS].mean().reindex(METHOD_ORDER).reset_index()
    average.insert(1, "display_method", average["method"].map(DISPLAY_NAMES))
    average.insert(2, "n_datasets", 8)
    return average


def build_by_schema(detail: pd.DataFrame):
    by_schema = (
        detail.groupby(["method", "schema"], sort=False)[METRICS]
        .mean()
        .reset_index()
    )
    by_schema.insert(1, "display_method", by_schema["method"].map(DISPLAY_NAMES))
    by_schema.insert(3, "n_datasets", 4)
    return by_schema


def build_all_in_one(detail: pd.DataFrame, by_schema: pd.DataFrame, average: pd.DataFrame):
    dataset_rows = detail.copy()
    dataset_rows.insert(0, "row_type", "dataset")
    dataset_rows.insert(5, "n_datasets", 1)

    schema_rows = by_schema.copy()
    schema_rows.insert(0, "row_type", "schema_average")
    schema_rows.insert(4, "dataset", "ALL")
    schema_rows.insert(5, "seed", "")
    schema_rows.insert(6, "steps", "")
    schema_rows.insert(7, "episode_reward", "")
    schema_rows.insert(8, "source_path", "")

    average_rows = average.copy()
    average_rows.insert(0, "row_type", "all_average")
    average_rows.insert(3, "schema", "ALL")
    average_rows.insert(4, "dataset", "ALL")
    average_rows.insert(5, "seed", "")
    average_rows.insert(7, "steps", "")
    average_rows.insert(8, "episode_reward", "")
    average_rows.insert(9, "source_path", "")

    columns = [
        "row_type",
        "method",
        "display_method",
        "schema",
        "dataset",
        "seed",
        "n_datasets",
        "steps",
        "episode_reward",
        "source_path",
        *METRICS,
    ]
    return pd.concat([dataset_rows[columns], schema_rows[columns], average_rows[columns]], ignore_index=True)


def format_table(average: pd.DataFrame):
    table = average[["display_method", "n_datasets", *METRICS]].copy()
    table = table.rename(columns={"display_method": "Method", "n_datasets": "N"})
    for metric in METRICS:
        best_value = table[metric].max()
        table[metric] = table[metric].map(
            lambda value: f"**{value:.3f}**" if abs(float(value) - float(best_value)) < 1e-12 else f"{value:.3f}"
        )
    return table


def render_table_png(table: pd.DataFrame, output_path: Path):
    plot_table = table.copy()
    best_cells = set()
    for col_idx, metric in enumerate(METRICS, start=2):
        for row_idx, value in enumerate(plot_table[metric]):
            if str(value).startswith("**") and str(value).endswith("**"):
                best_cells.add((row_idx + 1, col_idx))
                plot_table.at[row_idx, metric] = str(value).replace("**", "")

    fig_height = max(2.8, 0.45 * (len(plot_table) + 1))
    fig, ax = plt.subplots(figsize=(10.8, fig_height))
    ax.axis("off")
    table_artist = ax.table(
        cellText=plot_table.values.tolist(),
        colLabels=plot_table.columns.tolist(),
        cellLoc="center",
        loc="center",
    )
    table_artist.auto_set_font_size(False)
    table_artist.set_fontsize(9)
    table_artist.scale(1.0, 1.35)

    for (row_idx, col_idx), cell in table_artist.get_celld().items():
        if row_idx == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#d9ead3")
        elif (row_idx, col_idx) in best_cells:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#fff2cc")
        else:
            cell.set_facecolor("#ffffff")

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_package(package_path: Path, paths):
    if package_path.exists():
        package_path.unlink()
    with zipfile.ZipFile(package_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in paths:
            zf.write(path, arcname=path.name)


def safe_float(value):
    if value in ("", None):
        return ""
    try:
        return float(value)
    except (TypeError, ValueError):
        return ""


def safe_int(value):
    if value in ("", None):
        return ""
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return ""


def parse_steps_from_path(path: str):
    match = re.search(r"(\d+)_finish", str(path))
    return int(match.group(1)) if match else ""


if __name__ == "__main__":
    main()
