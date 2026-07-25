from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "covertype-exploration" / "outputs"
FINAL = ROOT / "outputs" / "final_results" / "covertype"
MAX_EP = 1000
SEEDS = (1, 2, 3)

LINE_W, LINE_H = 1920, 690
OP_W, OP_H = 1920, 1460
BG = (255, 255, 255)
BLACK = (0, 0, 0)
GRID = (214, 214, 214)


METHODS = [
    {
        "label": "MIRA",
        "color": "#4369ee",
        "reward_files": [
            OUT / "MIRA" / f"mira_seed{s}_v5_1" / f"mira_seed{s}_v5_1_mira_rewards.csv"
            for s in SEEDS
        ],
        "trace_files": [
            OUT
            / "MIRA"
            / f"mira_seed{s}_v5_1"
            / f"mira_seed{s}_v5_1_mira_exploration_trace.csv"
            for s in SEEDS
        ],
    },
    {
        "label": "Pure A3C",
        "color": "#c78325",
        "reward_files": [
            OUT / "A3Cpure" / f"pure_a3c_seed{s}_full" / f"pure_a3c_seed{s}_full_pure_a3c_rewards.csv"
            for s in SEEDS
        ],
        "trace_files": [
            OUT
            / "A3Cpure"
            / f"pure_a3c_seed{s}_full"
            / f"pure_a3c_seed{s}_full_pure_a3c_exploration_trace.csv"
            for s in SEEDS
        ],
    },
    {
        "label": "ATENA-style",
        "color": "#9b35f0",
        "reward_files": [
            OUT / "ATENA" / f"atena_seed{s}_full" / f"atena_seed{s}_full_atena_rewards.csv"
            for s in SEEDS
        ],
        "trace_files": [
            OUT / "ATENA" / f"atena_seed{s}_full" / f"atena_seed{s}_full_atena_exploration_trace.csv"
            for s in SEEDS
        ],
    },
    {
        "label": "ATENA-style + ext.",
        "color": "#ef5b2a",
        "reward_files": [
            OUT
            / "ATENA-EXT"
            / f"atena_ext_seed{s}_full"
            / f"atena_ext_seed{s}_full_atena_extrinsic_rewards.csv"
            for s in SEEDS
        ],
        "trace_files": [
            OUT
            / "ATENA-EXT"
            / f"atena_ext_seed{s}_full"
            / f"atena_ext_seed{s}_full_atena_extrinsic_exploration_trace.csv"
            for s in SEEDS
        ],
    },
    {
        "label": "DORA",
        "color": "#dd2f2f",
        "reward_files": [
            OUT / "DORA" / f"paper_a3c_seed{s}_full" / f"paper_a3c_seed{s}_full_paper_a3c_rewards.csv"
            for s in SEEDS
        ],
        "trace_files": [
            OUT
            / "DORA"
            / f"paper_a3c_seed{s}_full"
            / f"paper_a3c_seed{s}_full_paper_a3c_exploration_trace.csv"
            for s in SEEDS
        ],
    },
    {
        "label": "Random",
        "color": "#555555",
        "reward_files": [OUT / "random" / f"random_seed{s}_random_rewards.csv" for s in SEEDS],
        "trace_files": [OUT / "random" / f"random_seed{s}_random_exploration_trace.csv" for s in SEEDS],
    },
]

OP_CATS = [
    ("by_facet", "#1f77b4"),
    ("by_superset", "#ff7f0e"),
    ("by_neighbors", "#2ca02c"),
    ("by_distribution", "#d62728"),
]


def rgb(hex_color: str) -> tuple[int, int, int]:
    value = hex_color.lstrip("#")
    return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    path = Path(r"C:\Windows\Fonts\timesbd.ttf" if bold else r"C:\Windows\Fonts\times.ttf")
    if path.exists():
        return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


TITLE = font(42, True)
SUBTITLE = font(28, True)
AXIS = font(32)
TICK = font(24)
LEGEND = font(28)
OP_LEGEND = font(30)


def text_box(draw: ImageDraw.ImageDraw, text: str, fnt: ImageFont.ImageFont) -> tuple[int, int]:
    box = draw.textbbox((0, 0), text, font=fnt)
    return box[2] - box[0], box[3] - box[1]


def draw_center(draw: ImageDraw.ImageDraw, text: str, fnt: ImageFont.ImageFont, rect, fill=BLACK) -> None:
    x, y, w, h = rect
    tw, th = text_box(draw, text, fnt)
    draw.text((x + (w - tw) / 2, y + (h - th) / 2), text, font=fnt, fill=fill)


def draw_right(draw: ImageDraw.ImageDraw, text: str, fnt: ImageFont.ImageFont, rect, fill=BLACK) -> None:
    x, y, w, h = rect
    tw, th = text_box(draw, text, fnt)
    draw.text((x + w - tw, y + (h - th) / 2), text, font=fnt, fill=fill)


def draw_rotated_label(image: Image.Image, text: str, fnt: ImageFont.ImageFont, center: tuple[int, int]) -> None:
    temp = Image.new("RGBA", (640, 80), (255, 255, 255, 0))
    td = ImageDraw.Draw(temp)
    draw_center(td, text, fnt, (0, 0, 640, 80))
    rotated = temp.rotate(90, expand=True)
    image.paste(rotated, (center[0] - rotated.width // 2, center[1] - rotated.height // 2), rotated)


def valid_methods() -> list[dict]:
    methods = []
    for method in METHODS:
        reward_count = sum(path.exists() for path in method["reward_files"])
        trace_count = sum(path.exists() for path in method["trace_files"])
        print(f"{method['label']}: reward files {reward_count}/3, trace files {trace_count}/3")
        if reward_count:
            methods.append(method)
    return methods


def read_rewards(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "episode" not in df.columns:
        df.insert(0, "episode", np.arange(1, len(df) + 1))
    df["episode"] = pd.to_numeric(df["episode"], errors="coerce")
    df = df.dropna(subset=["episode"])
    df["episode"] = df["episode"].astype(int)
    df = df.sort_values("episode").drop_duplicates("episode", keep="last")
    return df[df["episode"].between(1, MAX_EP)].copy()


def seed_matrix(files: list[Path], metric: str) -> pd.DataFrame:
    cols = []
    for path in files:
        if not path.exists():
            continue
        df = read_rewards(path)
        if metric not in df.columns:
            print(f"missing {metric}: {path}")
            continue
        values = pd.to_numeric(df[metric], errors="coerce")
        cols.append(pd.Series(values.to_numpy(float), index=df["episode"].astype(int)))
    if not cols:
        return pd.DataFrame(index=range(1, MAX_EP + 1))
    return pd.concat(cols, axis=1).sort_index().reindex(range(1, MAX_EP + 1))


def nice_top(value: float) -> float:
    if not np.isfinite(value) or value <= 0:
        return 1.0
    mag = 10 ** math.floor(math.log10(value))
    norm = value / mag
    if norm <= 1:
        top = 1
    elif norm <= 2:
        top = 2
    elif norm <= 5:
        top = 5
    else:
        top = 10
    return top * mag


def tick_label(value: float, top: float) -> str:
    if abs(value - top) < 1e-12 and abs(top - 10.0) < 1e-12:
        return "10.0"
    if abs(value) >= 1:
        return str(int(round(value))) if abs(value - round(value)) < 1e-9 else f"{value:.1f}"
    if value == 0:
        return "0"
    return f"{value:.3f}".rstrip("0").rstrip(".")


def xy_line(x: float, y: float, left: int, top: int, pw: int, ph: int, ymax: float) -> tuple[float, float]:
    return left + x / MAX_EP * pw, top + ph - max(0.0, min(y, ymax)) / ymax * ph


def plot_metric(methods: list[dict], metric: str, ylabel: str, title: str, filename: str) -> None:
    left, top, pw, ph = 235, 105, 1230, 495
    image = Image.new("RGB", (LINE_W, LINE_H), BG)
    draw = ImageDraw.Draw(image)
    draw.text((left, 35), title, font=TITLE, fill=BLACK)

    series = []
    max_for_axis = 0.0
    for method in methods:
        matrix = seed_matrix(method["reward_files"], metric)
        if matrix.empty or matrix.shape[1] == 0:
            continue
        mean = matrix.mean(axis=1, skipna=True)
        std = matrix.std(axis=1, skipna=True).fillna(0.0)
        upper = mean + std
        max_for_axis = max(max_for_axis, float(np.nanmax(upper.to_numpy(float))))
        series.append((method, mean, std))
    if not series:
        return

    ymax = nice_top(max_for_axis * 1.08)
    for i in range(6):
        value = ymax * i / 5
        y = top + ph - value / ymax * ph
        draw.line((left, y, left + pw, y), fill=GRID, width=1)
        draw_right(draw, tick_label(value, ymax), TICK, (0, y - 18, left - 18, 38), fill=(80, 80, 80))
    for x_value in (0, 200, 400, 600, 800, 1000):
        x = left + x_value / MAX_EP * pw
        draw_center(draw, str(x_value), TICK, (x - 50, top + ph + 18, 100, 36), fill=(80, 80, 80))

    draw.line((left, top, left, top + ph), fill=BLACK, width=2)
    draw.line((left, top + ph, left + pw, top + ph), fill=BLACK, width=2)

    rgba = image.convert("RGBA")
    overlay = Image.new("RGBA", rgba.size, (255, 255, 255, 0))
    od = ImageDraw.Draw(overlay)
    xs = np.arange(1, MAX_EP + 1)
    for method, mean, std in series:
        color = rgb(method["color"])
        upper = (mean + std).ffill().fillna(0.0).to_numpy(float)
        lower = (mean - std).clip(lower=0.0).ffill().fillna(0.0).to_numpy(float)
        upper_points = [xy_line(float(x), float(y), left, top, pw, ph, ymax) for x, y in zip(xs, upper)]
        lower_points = [xy_line(float(x), float(y), left, top, pw, ph, ymax) for x, y in zip(xs[::-1], lower[::-1])]
        od.polygon(upper_points + lower_points, fill=color + (34,))
    rgba = Image.alpha_composite(rgba, overlay)
    image = rgba.convert("RGB")
    draw = ImageDraw.Draw(image)

    for method, mean, _std in series:
        color = rgb(method["color"])
        values = mean.ffill().fillna(0.0).to_numpy(float)
        points = [xy_line(float(x), float(y), left, top, pw, ph, ymax) for x, y in zip(xs, values)]
        draw.line(points, fill=color, width=3)

    draw_center(draw, "episode", AXIS, (left, top + ph + 72, pw, 52))
    draw_rotated_label(image, ylabel, AXIS, (65, top + ph // 2))

    legend_x, legend_y = 1498, 100
    legend_w, row_h = 390, 38
    legend_h = 34 + row_h * len(series)
    draw.rectangle((legend_x, legend_y, legend_x + legend_w, legend_y + legend_h), outline=(205, 205, 205), fill=BG)
    for idx, (method, _mean, _std) in enumerate(series):
        y = legend_y + 28 + idx * row_h
        draw.line((legend_x + 16, y, legend_x + 60, y), fill=rgb(method["color"]), width=4)
        draw.text((legend_x + 76, y - 18), method["label"], font=LEGEND, fill=BLACK)

    out = FINAL / filename
    image.save(out)
    print(f"Saved {out}")


def normalize_operator(value: object) -> str:
    text = str(value).lower()
    if "facet" in text:
        return "by_facet"
    if "superset" in text:
        return "by_superset"
    if "neighbor" in text:
        return "by_neighbors"
    if "distribution" in text:
        return "by_distribution"
    return "by_distribution"


@lru_cache(maxsize=None)
def trace_episode_counts(path_text: str) -> pd.DataFrame:
    path = Path(path_text)
    if not path.exists():
        return pd.DataFrame(0.0, index=range(1, MAX_EP + 1), columns=[name for name, _ in OP_CATS])
    df = pd.read_csv(path, usecols=lambda col: col in {"episode", "operator"})
    if "episode" not in df.columns or "operator" not in df.columns:
        return pd.DataFrame(0.0, index=range(1, MAX_EP + 1), columns=[name for name, _ in OP_CATS])
    df["episode"] = pd.to_numeric(df["episode"], errors="coerce")
    df = df.dropna(subset=["episode"])
    df["episode"] = df["episode"].astype(int)
    df = df[df["episode"].between(1, MAX_EP)]
    df["family"] = df["operator"].map(normalize_operator)
    counts = df.groupby(["episode", "family"]).size().unstack().fillna(0.0)
    counts = counts.reindex(range(1, MAX_EP + 1), fill_value=0.0)
    for name, _color in OP_CATS:
        if name not in counts.columns:
            counts[name] = 0.0
    return counts[[name for name, _color in OP_CATS]].astype(float)


def method_operator_counts(method: dict) -> pd.DataFrame:
    frames = [trace_episode_counts(str(path)) for path in method["trace_files"] if path.exists()]
    if not frames:
        return pd.DataFrame(0.0, index=range(1, MAX_EP + 1), columns=[name for name, _ in OP_CATS])
    return sum(frames) / float(len(frames))


def xy_op(x: float, y: float, left: int, top: int, pw: int, ph: int) -> tuple[float, float]:
    return left + x / MAX_EP * pw, top + ph - max(0.0, min(y, 500.0)) / 500.0 * ph


def draw_operator_panel(
    draw: ImageDraw.ImageDraw,
    overlay: ImageDraw.ImageDraw,
    method: dict,
    rect: tuple[int, int, int, int],
) -> None:
    left, top, pw, ph = rect
    draw.text((left, top - 31), method["label"], font=SUBTITLE, fill=BLACK)
    for value in (0, 100, 200, 300, 400, 500):
        y = top + ph - value / 500.0 * ph
        draw.line((left, y, left + pw, y), fill=GRID, width=1)
        draw_right(draw, str(value), TICK, (left - 76, y - 17, 66, 34), fill=(90, 90, 90))
    for x_value in (0, 250, 500, 750, 1000):
        x = left + x_value / MAX_EP * pw
        draw_center(draw, str(x_value), TICK, (x - 45, top + ph + 8, 90, 32), fill=(90, 90, 90))
    draw.line((left, top, left, top + ph), fill=BLACK, width=2)
    draw.line((left, top + ph, left + pw, top + ph), fill=BLACK, width=2)

    data = method_operator_counts(method)
    xs = np.arange(1, MAX_EP + 1)
    bottom = np.zeros(MAX_EP, dtype=float)
    for name, color_hex in OP_CATS:
        vals = data[name].to_numpy(float)
        top_vals = bottom + vals
        top_points = [xy_op(float(x), float(y), left, top, pw, ph) for x, y in zip(xs, top_vals)]
        bottom_points = [xy_op(float(x), float(y), left, top, pw, ph) for x, y in zip(xs[::-1], bottom[::-1])]
        overlay.polygon(top_points + bottom_points, fill=rgb(color_hex) + (255,))
        bottom = top_vals


def plot_operator_evolution(methods: list[dict]) -> None:
    image = Image.new("RGB", (OP_W, OP_H), BG)
    draw = ImageDraw.Draw(image)
    draw.text((92, 42), "Covertype Operator Distribution Evolution", font=TITLE, fill=BLACK)

    rgba = image.convert("RGBA")
    overlay = Image.new("RGBA", rgba.size, (255, 255, 255, 0))
    od = ImageDraw.Draw(overlay)
    panel_w, panel_h = 670, 300
    xs = [130, 890]
    ys = [130, 510, 890]
    for idx, method in enumerate(methods):
        row, col = divmod(idx, 2)
        if row >= len(ys):
            break
        draw_operator_panel(draw, od, method, (xs[col], ys[row], panel_w, panel_h))

    rgba = Image.alpha_composite(rgba, overlay)
    image = rgba.convert("RGB")
    draw = ImageDraw.Draw(image)
    legend_x, legend_y = 1582, 130
    draw.rectangle((legend_x, legend_y, legend_x + 300, legend_y + 205), outline=(205, 205, 205), fill=BG)
    for idx, (name, color_hex) in enumerate(OP_CATS):
        y = legend_y + 32 + idx * 39
        draw.rectangle((legend_x + 14, y - 13, legend_x + 46, y + 13), fill=rgb(color_hex))
        draw.text((legend_x + 66, y - 20), name, font=OP_LEGEND, fill=BLACK)

    out = FINAL / "covertype_operator_distribution_evolution.png"
    image.save(out)
    print(f"Saved {out}")


def main() -> None:
    FINAL.mkdir(parents=True, exist_ok=True)
    methods = valid_methods()
    plot_metric(methods, "extrinsic_reward", "Extrinsic Reward", "Covertype Extrinsic Reward", "covertype_extrinsic_reward.png")
    plot_metric(
        methods,
        "cumulative_extrinsic_reward",
        "Cumulative Extrinsic Reward",
        "Covertype Cumulative Extrinsic Reward",
        "covertype_cumulative_extrinsic_reward.png",
    )
    plot_metric(
        methods,
        "cumulative_unique_sets_viewed",
        "Cumulative Unique Sets Viewed",
        "Covertype Cumulative Unique Sets Viewed",
        "covertype_cumulative_unique_sets_viewed.png",
    )
    plot_metric(methods, "target_efficiency", "Target Efficiency", "Covertype Target Efficiency", "covertype_target_efficiency.png")
    plot_metric(
        methods,
        "cumulative_target_efficiency",
        "Cumulative Target Efficiency",
        "Covertype Cumulative Target Efficiency",
        "covertype_cumulative_target_efficiency.png",
    )
    plot_operator_evolution(methods)


if __name__ == "__main__":
    main()


