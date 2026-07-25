from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
BASE = ROOT / "covertype-exploration" / "outputs"
FINAL_DIR = ROOT / "outputs" / "final_results" / "covertype"
MAX_EPISODE = 1000
ROLLING_WINDOW = 10

FONT_REG = Path(r"C:\Windows\Fonts\times.ttf")
FONT_BOLD = Path(r"C:\Windows\Fonts\timesbd.ttf")

LINE_SIZE = (1720, 620)
OP_SIZE = (1720, 1480)
LINE_PLOT = {"left": 210, "right": 410, "top": 88, "bottom": 88}
OP_PLOT = {"left": 118, "right": 306, "top": 112, "bottom": 102}


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONT_BOLD if bold else FONT_REG), size=size)


TITLE_FONT = font(38, True)
AXIS_FONT = font(29)
TICK_FONT = font(20)
LEGEND_FONT = font(24)
SUBTITLE_FONT = font(24, True)
OP_LEGEND_FONT = font(24)


METHODS = [
    {
        "label": "MIRA",
        "color": "#3f63e8",
        "reward_files": [
            BASE / "MIRA" / f"mira_seed{seed}_v5_1" / f"mira_seed{seed}_v5_1_mira_rewards.csv"
            for seed in (1, 2, 3)
        ],
        "trace_files": [
            BASE / "MIRA" / f"mira_seed{seed}_v5_1" / f"mira_seed{seed}_v5_1_mira_exploration_trace.csv"
            for seed in (1, 2, 3)
        ],
    },
    {
        "label": "Pure A3C",
        "color": "#c47c25",
        "reward_files": [
            BASE / "A3Cpure" / f"pure_a3c_seed{seed}_full" / f"pure_a3c_seed{seed}_full_pure_a3c_rewards.csv"
            for seed in (1, 2, 3)
        ],
        "trace_files": [
            BASE / "A3Cpure" / f"pure_a3c_seed{seed}_full" / f"pure_a3c_seed{seed}_full_pure_a3c_exploration_trace.csv"
            for seed in (1, 2, 3)
        ],
    },
    {
        "label": "ATENA-style",
        "color": "#9b35ef",
        "reward_files": [
            BASE / "ATENA" / f"atena_seed{seed}_full" / f"atena_seed{seed}_full_atena_rewards.csv"
            for seed in (1, 2, 3)
        ],
        "trace_files": [
            BASE / "ATENA" / f"atena_seed{seed}_full" / f"atena_seed{seed}_full_atena_exploration_trace.csv"
            for seed in (1, 2, 3)
        ],
    },
    {
        "label": "ATENA-style + ext.",
        "color": "#e95a2a",
        "reward_files": [
            BASE / "ATENA-EXT" / f"atena_ext_seed{seed}_full" / f"atena_ext_seed{seed}_full_atena_extrinsic_rewards.csv"
            for seed in (1, 2, 3)
        ],
        "trace_files": [
            BASE / "ATENA-EXT" / f"atena_ext_seed{seed}_full" / f"atena_ext_seed{seed}_full_atena_extrinsic_exploration_trace.csv"
            for seed in (1, 2, 3)
        ],
    },
    {
        "label": "DORA",
        "color": "#df2d2f",
        "reward_files": [
            BASE / "DORA" / f"paper_a3c_seed{seed}_full" / f"paper_a3c_seed{seed}_full_paper_a3c_rewards.csv"
            for seed in (1, 2, 3)
        ],
        "trace_files": [
            BASE / "DORA" / f"paper_a3c_seed{seed}_full" / f"paper_a3c_seed{seed}_full_paper_a3c_exploration_trace.csv"
            for seed in (1, 2, 3)
        ],
    },
    {
        "label": "Random",
        "color": "#4f4f4f",
        "reward_files": [
            BASE / "random" / f"random_seed{seed}_random_rewards.csv" for seed in (1, 2, 3)
        ],
        "trace_files": [
            BASE / "random" / f"random_seed{seed}_random_exploration_trace.csv" for seed in (1, 2, 3)
        ],
    },
]

OPERATOR_ORDER = ["by_facet", "by_superset", "by_neighbors", "by_distribution"]
OPERATOR_COLORS = {
    "by_facet": "#1f77b4",
    "by_superset": "#ff7f0e",
    "by_neighbors": "#2ca02c",
    "by_distribution": "#d62728",
}


def rgb(color: str, alpha: int = 255) -> tuple[int, int, int, int]:
    color = color.lstrip("#")
    return (int(color[0:2], 16), int(color[2:4], 16), int(color[4:6], 16), alpha)


def text_size(draw: ImageDraw.ImageDraw, text: str, fnt: ImageFont.FreeTypeFont) -> tuple[int, int]:
    box = draw.textbbox((0, 0), str(text), font=fnt)
    return box[2] - box[0], box[3] - box[1]


def draw_center(draw: ImageDraw.ImageDraw, xy: tuple[float, float], text: str, fnt: ImageFont.FreeTypeFont, fill=(20, 20, 20, 255)) -> None:
    w, h = text_size(draw, text, fnt)
    draw.text((xy[0] - w / 2, xy[1] - h / 2), text, font=fnt, fill=fill)


def draw_right(draw: ImageDraw.ImageDraw, x: float, y: float, text: str, fnt: ImageFont.FreeTypeFont, fill=(80, 80, 80, 255)) -> None:
    w, h = text_size(draw, text, fnt)
    draw.text((x - w, y - h / 2), text, font=fnt, fill=fill)


def draw_rotated_label(img: Image.Image, text: str, center: tuple[float, float]) -> None:
    tmp = Image.new("RGBA", (800, 70), (255, 255, 255, 0))
    d = ImageDraw.Draw(tmp)
    draw_center(d, (400, 35), text, AXIS_FONT)
    rot = tmp.rotate(90, expand=True)
    img.alpha_composite(rot, (int(center[0] - rot.width / 2), int(center[1] - rot.height / 2)))


def read_reward_file(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    if "episode" not in df.columns:
        df.insert(0, "episode", np.arange(1, len(df) + 1))
    df["episode"] = pd.to_numeric(df["episode"], errors="coerce")
    df = df.dropna(subset=["episode"])
    df["episode"] = df["episode"].astype(int)
    df = df.sort_values("episode").drop_duplicates("episode", keep="last")
    return df[df["episode"].between(1, MAX_EPISODE)].copy()


def seed_metric_matrix(files: list[Path], metric: str) -> tuple[np.ndarray, np.ndarray]:
    series = []
    for path in files:
        df = read_reward_file(path)
        if metric not in df.columns:
            raise KeyError(f"{metric} missing in {path}")
        values = pd.to_numeric(df[metric], errors="coerce")
        series.append(pd.Series(values.to_numpy(dtype=float), index=df["episode"].astype(int)))
    aligned = pd.concat(series, axis=1).sort_index().reindex(range(1, MAX_EPISODE + 1))
    return aligned.index.to_numpy(dtype=int), aligned.to_numpy(dtype=float)


def smooth(values: np.ndarray, window: int = ROLLING_WINDOW) -> np.ndarray:
    return pd.Series(values).rolling(window=window, min_periods=1, center=True).mean().to_numpy(dtype=float)


def nice_ticks(vmin: float, vmax: float, count: int = 5) -> list[float]:
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        return [0.0, 1.0]
    raw = (vmax - vmin) / max(count - 1, 1)
    mag = 10 ** math.floor(math.log10(abs(raw))) if raw else 1
    candidates = np.array([1, 2, 2.5, 5, 10], dtype=float) * mag
    step = float(candidates[np.argmin(np.abs(candidates - raw))])
    lo = math.floor(vmin / step) * step
    hi = math.ceil(vmax / step) * step
    ticks = []
    cur = lo
    while cur <= hi + step * 0.5:
        ticks.append(cur)
        cur += step
    if len(ticks) > 7:
        ticks = ticks[::2]
    return ticks


def fmt_tick(value: float) -> str:
    if abs(value) >= 1000000:
        return f"{value / 1000000:.1f}M"
    if abs(value) >= 10000:
        return f"{value:.0f}"
    if abs(value) >= 1000:
        return f"{value:.0f}"
    if abs(value) >= 10:
        return f"{value:.0f}"
    if abs(value - round(value)) < 1e-9:
        return f"{value:.0f}"
    return f"{value:.1f}"


def line_points(x: np.ndarray, y: np.ndarray, sx, sy) -> list[tuple[float, float]]:
    return [(sx(float(xi)), sy(float(yi))) for xi, yi in zip(x, y) if np.isfinite(yi)]


def draw_polyline(draw: ImageDraw.ImageDraw, pts: list[tuple[float, float]], color: tuple[int, int, int, int], width: int) -> None:
    if len(pts) >= 2:
        draw.line(pts, fill=color, width=width, joint="curve")


def plot_metric(metric: str, ylabel: str, title: str, filename: str, ylim: tuple[float, float] | None = None) -> None:
    width, height = LINE_SIZE
    img = Image.new("RGBA", (width, height), (255, 255, 255, 255))
    draw = ImageDraw.Draw(img)
    left = LINE_PLOT["left"]
    right = width - LINE_PLOT["right"]
    top = LINE_PLOT["top"]
    bottom = height - LINE_PLOT["bottom"]
    plot_w = right - left
    plot_h = bottom - top

    all_rows = []
    ymax = 0.0
    for method in METHODS:
        x, matrix = seed_metric_matrix(method["reward_files"], metric)
        mean = smooth(np.nanmean(matrix, axis=1))
        std = smooth(np.nanstd(matrix, axis=1))
        lower = np.maximum(mean - std, 0.0)
        upper = mean + std
        all_rows.append((method, x, mean, lower, upper))
        if np.isfinite(upper).any():
            ymax = max(ymax, float(np.nanmax(upper)))

    if ylim is None:
        y0, y1 = 0.0, ymax * 1.05 if ymax > 0 else 1.0
    else:
        y0, y1 = ylim

    def sx(v: float) -> float:
        return left + (v / MAX_EPISODE) * plot_w

    def sy(v: float) -> float:
        return bottom - ((v - y0) / max(y1 - y0, 1e-9)) * plot_h

    draw.text((left, 28), title, font=TITLE_FONT, fill=(15, 15, 15, 255))
    for tick in nice_ticks(y0, y1, 5):
        yy = sy(tick)
        draw.line([(left, yy), (right, yy)], fill=(216, 216, 216, 255), width=1)
        draw_right(draw, left - 14, yy, fmt_tick(tick), TICK_FONT)
    for tick in [0, 200, 400, 600, 800, 1000]:
        xx = sx(tick)
        tw, th = text_size(draw, str(tick), TICK_FONT)
        draw.text((xx - tw / 2, bottom + 22), str(tick), font=TICK_FONT, fill=(20, 20, 20, 255))

    draw.line([(left, top), (left, bottom), (right, bottom)], fill=(0, 0, 0, 255), width=2)

    for method, x, mean, lower, upper in all_rows:
        upper_pts = line_points(x, upper, sx, sy)
        lower_pts = line_points(x, lower, sx, sy)
        if len(upper_pts) >= 2 and len(lower_pts) >= 2:
            polygon = upper_pts + list(reversed(lower_pts))
            draw.polygon(polygon, fill=rgb(method["color"], 22))
    for method, x, mean, lower, upper in all_rows:
        draw_polyline(draw, line_points(x, mean, sx, sy), rgb(method["color"], 255), 3)

    draw_center(draw, ((left + right) / 2, height - 24), "episode", AXIS_FONT)
    draw_rotated_label(img, ylabel, (58, (top + bottom) / 2))

    legend_x = right + 28
    legend_y = top - 4
    legend_w = width - legend_x - 30
    legend_h = 282
    draw.rectangle([legend_x, legend_y, legend_x + legend_w, legend_y + legend_h], outline=(214, 214, 214, 255), width=1, fill=(255, 255, 255, 255))
    for idx, (method, *_rest) in enumerate(all_rows):
        yy = legend_y + 28 + idx * 34
        draw.line([(legend_x + 14, yy + 10), (legend_x + 58, yy + 10)], fill=rgb(method["color"], 255), width=3)
        draw.text((legend_x + 72, yy - 4), method["label"], font=LEGEND_FONT, fill=(15, 15, 15, 255))

    img.convert("RGB").save(FINAL_DIR / filename, quality=95)


def normalize_operator(value: object) -> str | None:
    text = str(value).lower()
    if "facet" in text:
        return "by_facet"
    if "superset" in text:
        return "by_superset"
    if "neighbor" in text:
        return "by_neighbors"
    if "distribution" in text:
        return "by_distribution"
    return None


def seed_operator_counts(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, usecols=lambda col: col in {"episode", "operator"})
    if "episode" not in df.columns or "operator" not in df.columns:
        raise KeyError(f"episode/operator missing in {path}")
    df["episode"] = pd.to_numeric(df["episode"], errors="coerce")
    df = df.dropna(subset=["episode"])
    df["episode"] = df["episode"].astype(int)
    df = df[df["episode"].between(1, MAX_EPISODE)].copy()
    df["operator_family"] = df["operator"].map(normalize_operator)
    df = df.dropna(subset=["operator_family"])
    counts = (
        df.groupby(["episode", "operator_family"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=range(1, MAX_EPISODE + 1), columns=OPERATOR_ORDER, fill_value=0)
    )
    return counts


def mean_operator_counts(files: list[Path]) -> pd.DataFrame:
    matrices = [seed_operator_counts(path) for path in files]
    stacked = np.stack([matrix.to_numpy(dtype=float) for matrix in matrices], axis=0)
    averaged = np.nanmean(stacked, axis=0)
    return pd.DataFrame(averaged, index=range(1, MAX_EPISODE + 1), columns=OPERATOR_ORDER)


def draw_stacked_area(ax_draw: ImageDraw.ImageDraw, x_values: np.ndarray, y_arrays: list[np.ndarray], box: tuple[int, int, int, int], y_max: float) -> None:
    left, top, right, bottom = box
    plot_w = right - left
    plot_h = bottom - top

    def sx(v: float) -> float:
        return left + (v / MAX_EPISODE) * plot_w

    def sy(v: float) -> float:
        return bottom - (v / max(y_max, 1e-9)) * plot_h

    baseline = np.zeros_like(x_values, dtype=float)
    for op_name, y in zip(OPERATOR_ORDER, y_arrays):
        top_line = baseline + y
        upper = [(sx(float(x)), sy(float(v))) for x, v in zip(x_values, top_line)]
        lower = [(sx(float(x)), sy(float(v))) for x, v in zip(x_values[::-1], baseline[::-1])]
        ax_draw.polygon(upper + lower, fill=rgb(OPERATOR_COLORS[op_name], 255))
        baseline = top_line


def plot_operator_distribution() -> None:
    width, height = OP_SIZE
    img = Image.new("RGBA", (width, height), (255, 255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((116, 32), "Covertype Operator Distribution Evolution", font=TITLE_FONT, fill=(15, 15, 15, 255))

    panel_w = 600
    panel_h = 260
    x_gap = 78
    y_gap = 72
    start_x = 116
    start_y = 108
    positions = [
        (start_x, start_y),
        (start_x + panel_w + x_gap, start_y),
        (start_x, start_y + panel_h + y_gap),
        (start_x + panel_w + x_gap, start_y + panel_h + y_gap),
        (start_x, start_y + 2 * (panel_h + y_gap)),
        (start_x + panel_w + x_gap, start_y + 2 * (panel_h + y_gap)),
    ]
    x_values = np.arange(1, MAX_EPISODE + 1)
    y_max = 500.0

    for idx, method in enumerate(METHODS):
        x0, y0 = positions[idx]
        left, top, right, bottom = x0, y0 + 32, x0 + panel_w, y0 + panel_h
        counts = mean_operator_counts(method["trace_files"])
        ys = [smooth(counts[col].to_numpy(dtype=float)) for col in OPERATOR_ORDER]

        draw.text((x0, y0), method["label"], font=SUBTITLE_FONT, fill=(15, 15, 15, 255))
        for tick in [0, 100, 200, 300, 400, 500]:
            yy = bottom - (tick / y_max) * (bottom - top)
            draw.line([(left, yy), (right, yy)], fill=(224, 224, 224, 255), width=1)
            draw_right(draw, left - 8, yy, str(tick), TICK_FONT)
        for tick in [0, 250, 500, 750, 1000]:
            xx = left + (tick / MAX_EPISODE) * (right - left)
            tw, th = text_size(draw, str(tick), TICK_FONT)
            draw.text((xx - tw / 2, bottom + 10), str(tick), font=TICK_FONT, fill=(90, 90, 90, 255))
        draw.line([(left, top), (left, bottom), (right, bottom)], fill=(0, 0, 0, 255), width=2)
        draw_stacked_area(draw, x_values, ys, (left, top, right, bottom), y_max)

    legend_x = 1415
    legend_y = 110
    legend_w = 258
    legend_h = 180
    draw.rectangle([legend_x, legend_y, legend_x + legend_w, legend_y + legend_h], outline=(214, 214, 214, 255), width=1, fill=(255, 255, 255, 255))
    for i, op in enumerate(OPERATOR_ORDER):
        yy = legend_y + 22 + i * 38
        draw.rectangle([legend_x + 18, yy, legend_x + 48, yy + 18], fill=rgb(OPERATOR_COLORS[op], 255))
        draw.text((legend_x + 60, yy - 6), op, font=OP_LEGEND_FONT, fill=(15, 15, 15, 255))

    img.convert("RGB").save(FINAL_DIR / "covertype_operator_distribution_evolution.png", quality=95)


def main() -> None:
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    plot_metric(
        "cumulative_extrinsic_reward",
        "Cumulative Extrinsic Reward",
        "Covertype Cumulative Extrinsic Reward",
        "covertype_cumulative_extrinsic_reward.png",
    )
    plot_metric(
        "cumulative_target_efficiency",
        "Cumulative Target Efficiency",
        "Covertype Cumulative Target Efficiency",
        "covertype_cumulative_target_efficiency.png",
        ylim=(0, 10),
    )
    plot_metric(
        "cumulative_unique_sets_viewed",
        "Cumulative Unique Sets Viewed",
        "Covertype Cumulative Unique Sets Viewed",
        "covertype_cumulative_unique_sets_viewed.png",
    )
    plot_metric(
        "extrinsic_reward",
        "Extrinsic Reward",
        "Covertype Extrinsic Reward",
        "covertype_extrinsic_reward.png",
    )
    plot_metric(
        "target_efficiency",
        "Target Efficiency",
        "Covertype Target Efficiency",
        "covertype_target_efficiency.png",
        ylim=(0, 10),
    )
    plot_operator_distribution()
    for path in sorted(FINAL_DIR.glob("covertype_*.png")):
        print(path)


if __name__ == "__main__":
    main()
