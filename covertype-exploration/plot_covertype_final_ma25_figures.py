from pathlib import Path
import csv
from collections import defaultdict
import numpy as np
from PIL import Image, ImageDraw, ImageFont

WINDOW = 25
ROOT = Path(__file__).resolve().parents[1]
COVERTYPE_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = COVERTYPE_ROOT / "outputs"
FINAL_DIR = ROOT / "outputs" / "final_results" / "covertype"
FINAL_DIR.mkdir(parents=True, exist_ok=True)

METHOD_ORDER = ["MIRA", "MIRA (no ext.)", "Pure A3C", "ATENA-style", "ATENA-style + ext.", "DORA", "Random"]
LINE_COLORS = {
    "MIRA": (65, 105, 225),
    "MIRA (no ext.)": (44, 125, 118),
    "Pure A3C": (194, 126, 34),
    "ATENA-style": (154, 47, 219),
    "ATENA-style + ext.": (229, 91, 45),
    "DORA": (213, 58, 54),
    "Random": (82, 82, 82),
}
FILL_ALPHA = {"MIRA": 42, "MIRA (no ext.)": 30, "Pure A3C": 28, "ATENA-style": 24, "ATENA-style + ext.": 30, "DORA": 30, "Random": 22}
OPERATOR_ORDER = ["by_facet", "by_superset", "by_neighbors", "by_distribution"]
OPERATOR_COLORS = {
    "by_facet": (31, 119, 180),
    "by_superset": (255, 127, 14),
    "by_neighbors": (44, 160, 44),
    "by_distribution": (214, 39, 40),
}

REWARD_FILES = {
    "MIRA": {
        1: OUTPUT_ROOT / "MIRA" / "mira_seed1_v6" / "mira_seed1_v6_mira_rewards.csv",
        2: OUTPUT_ROOT / "MIRA" / "mira_seed2_v6" / "mira_seed2_v6_mira_rewards.csv",
        3: OUTPUT_ROOT / "MIRA" / "mira_seed3_v6" / "mira_seed3_v6_mira_rewards.csv",
    },
    "MIRA (no ext.)": {
        1: OUTPUT_ROOT / "MIRA_noEXT" / "mira_no_ext_seed1_v6" / "mira_no_ext_seed1_v6_mira_no_ext_rewards.csv",
        2: OUTPUT_ROOT / "MIRA_noEXT" / "mira_no_ext_seed2_v6" / "mira_no_ext_seed2_v6_mira_no_ext_rewards.csv",
        3: OUTPUT_ROOT / "MIRA_noEXT" / "mira_no_ext_seed3_v6" / "mira_no_ext_seed3_v6_mira_no_ext_rewards.csv",
    },
    "Pure A3C": {
        1: OUTPUT_ROOT / "A3Cpure" / "pure_a3c_seed1_full" / "pure_a3c_seed1_full_pure_a3c_rewards.csv",
        2: OUTPUT_ROOT / "A3Cpure" / "pure_a3c_seed2_full" / "pure_a3c_seed2_full_pure_a3c_rewards.csv",
        3: OUTPUT_ROOT / "A3Cpure" / "pure_a3c_seed3_full" / "pure_a3c_seed3_full_pure_a3c_rewards.csv",
    },
    "ATENA-style": {
        1: OUTPUT_ROOT / "ATENA" / "atena_seed1_full" / "atena_seed1_full_atena_rewards.csv",
        2: OUTPUT_ROOT / "ATENA" / "atena_seed2_full" / "atena_seed2_full_atena_rewards.csv",
        3: OUTPUT_ROOT / "ATENA" / "atena_seed3_full" / "atena_seed3_full_atena_rewards.csv",
    },
    "ATENA-style + ext.": {
        1: OUTPUT_ROOT / "ATENA-EXT" / "atena_ext_seed1_full" / "atena_ext_seed1_full_atena_extrinsic_rewards.csv",
        2: OUTPUT_ROOT / "ATENA-EXT" / "atena_ext_seed2_full" / "atena_ext_seed2_full_atena_extrinsic_rewards.csv",
        3: OUTPUT_ROOT / "ATENA-EXT" / "atena_ext_seed3_full" / "atena_ext_seed3_full_atena_extrinsic_rewards.csv",
    },
    "DORA": {
        1: OUTPUT_ROOT / "DORA" / "paper_a3c_seed1_full" / "paper_a3c_seed1_full_paper_a3c_rewards.csv",
        2: OUTPUT_ROOT / "DORA" / "paper_a3c_seed2_full" / "paper_a3c_seed2_full_paper_a3c_rewards.csv",
        3: OUTPUT_ROOT / "DORA" / "paper_a3c_seed3_full" / "paper_a3c_seed3_full_paper_a3c_rewards.csv",
    },
    "Random": {
        1: OUTPUT_ROOT / "random" / "random_seed1_random_rewards.csv",
        2: OUTPUT_ROOT / "random" / "random_seed2_random_rewards.csv",
        3: OUTPUT_ROOT / "random" / "random_seed3_random_rewards.csv",
    },
}

TRACE_FILES = {
    "MIRA": {
        1: OUTPUT_ROOT / "MIRA" / "mira_seed1_v6" / "mira_seed1_v6_mira_exploration_trace.csv",
        2: OUTPUT_ROOT / "MIRA" / "mira_seed2_v6" / "mira_seed2_v6_mira_exploration_trace.csv",
        3: OUTPUT_ROOT / "MIRA" / "mira_seed3_v6" / "mira_seed3_v6_mira_exploration_trace.csv",
    },
    "MIRA (no ext.)": {
        1: OUTPUT_ROOT / "MIRA_noEXT" / "mira_no_ext_seed1_v6" / "mira_no_ext_seed1_v6_mira_no_ext_exploration_trace.csv",
        2: OUTPUT_ROOT / "MIRA_noEXT" / "mira_no_ext_seed2_v6" / "mira_no_ext_seed2_v6_mira_no_ext_exploration_trace.csv",
        3: OUTPUT_ROOT / "MIRA_noEXT" / "mira_no_ext_seed3_v6" / "mira_no_ext_seed3_v6_mira_no_ext_exploration_trace.csv",
    },
    "Pure A3C": {
        1: OUTPUT_ROOT / "A3Cpure" / "pure_a3c_seed1_full" / "pure_a3c_seed1_full_pure_a3c_exploration_trace.csv",
        2: OUTPUT_ROOT / "A3Cpure" / "pure_a3c_seed2_full" / "pure_a3c_seed2_full_pure_a3c_exploration_trace.csv",
        3: OUTPUT_ROOT / "A3Cpure" / "pure_a3c_seed3_full" / "pure_a3c_seed3_full_pure_a3c_exploration_trace.csv",
    },
    "ATENA-style": {
        1: OUTPUT_ROOT / "ATENA" / "atena_seed1_full" / "atena_seed1_full_atena_exploration_trace.csv",
        2: OUTPUT_ROOT / "ATENA" / "atena_seed2_full" / "atena_seed2_full_atena_exploration_trace.csv",
        3: OUTPUT_ROOT / "ATENA" / "atena_seed3_full" / "atena_seed3_full_atena_exploration_trace.csv",
    },
    "ATENA-style + ext.": {
        1: OUTPUT_ROOT / "ATENA-EXT" / "atena_ext_seed1_full" / "atena_ext_seed1_full_atena_extrinsic_exploration_trace.csv",
        2: OUTPUT_ROOT / "ATENA-EXT" / "atena_ext_seed2_full" / "atena_ext_seed2_full_atena_extrinsic_exploration_trace.csv",
        3: OUTPUT_ROOT / "ATENA-EXT" / "atena_ext_seed3_full" / "atena_ext_seed3_full_atena_extrinsic_exploration_trace.csv",
    },
    "DORA": {
        1: OUTPUT_ROOT / "DORA" / "paper_a3c_seed1_full" / "paper_a3c_seed1_full_paper_a3c_exploration_trace.csv",
        2: OUTPUT_ROOT / "DORA" / "paper_a3c_seed2_full" / "paper_a3c_seed2_full_paper_a3c_exploration_trace.csv",
        3: OUTPUT_ROOT / "DORA" / "paper_a3c_seed3_full" / "paper_a3c_seed3_full_paper_a3c_exploration_trace.csv",
    },
    "Random": {
        1: OUTPUT_ROOT / "random" / "random_seed1_random_exploration_trace.csv",
        2: OUTPUT_ROOT / "random" / "random_seed2_random_exploration_trace.csv",
        3: OUTPUT_ROOT / "random" / "random_seed3_random_exploration_trace.csv",
    },
}

FONT_CANDIDATES = {
    "regular": [r"C:\Windows\Fonts\times.ttf", r"C:\Windows\Fonts\timesnewroman.ttf", r"C:\Windows\Fonts\arial.ttf"],
    "bold": [r"C:\Windows\Fonts\timesbd.ttf", r"C:\Windows\Fonts\timesnewromanbd.ttf", r"C:\Windows\Fonts\arialbd.ttf"],
}

def load_font(kind, size):
    for fp in FONT_CANDIDATES[kind]:
        if Path(fp).exists():
            return ImageFont.truetype(fp, size)
    return ImageFont.load_default()

def moving_average(values, window=WINDOW):
    values = np.asarray(values, dtype=float)
    out = np.empty_like(values, dtype=float)
    csum = np.cumsum(np.insert(values, 0, 0.0))
    for i in range(len(values)):
        start = max(0, i - window + 1)
        out[i] = (csum[i + 1] - csum[start]) / (i - start + 1)
    return out

def read_metric(path, metric):
    rows = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if metric not in reader.fieldnames:
            raise KeyError(f"{metric} not found in {path}")
        for row in reader:
            rows[int(float(row["episode"]))] = float(row[metric])
    return rows

def compute_line_stats(metric):
    result = {}
    for method, seed_paths in REWARD_FILES.items():
        seed_series = {seed: read_metric(path, metric) for seed, path in seed_paths.items()}
        common = sorted(set.intersection(*(set(s.keys()) for s in seed_series.values())))
        arr = []
        for seed in sorted(seed_series):
            vals = np.asarray([seed_series[seed][ep] for ep in common], dtype=float)
            arr.append(moving_average(vals))
        arr = np.vstack(arr).T
        result[method] = {"episodes": np.asarray(common, dtype=float), "mean": arr.mean(axis=1), "std": arr.std(axis=1)}
    common_all = sorted(set.intersection(*(set(map(int, d["episodes"])) for d in result.values())))
    for method, d in result.items():
        idx_by_ep = {int(ep): idx for idx, ep in enumerate(d["episodes"])}
        idx = [idx_by_ep[ep] for ep in common_all]
        d["episodes"] = np.asarray(common_all, dtype=float)
        d["mean"] = d["mean"][idx]
        d["std"] = d["std"][idx]
    return result

def fmt_tick(v, decimals=False):
    v = float(v)
    if decimals:
        return "0.00" if abs(v) < 1e-12 else (f"{v:.2f}" if v < 1 else f"{v:.1f}")
    return str(int(round(v)))

def draw_centered(draw, xy, text, font, fill):
    x, y = xy
    bbox = draw.textbbox((0, 0), text, font=font)
    draw.text((x - (bbox[2] - bbox[0]) / 2, y - (bbox[3] - bbox[1]) / 2), text, font=font, fill=fill)

def plot_line_metric(metric, title, ylabel, filename, y_max=None, y_ticks=None, decimals=False):
    data = compute_line_stats(metric)
    if y_max is None:
        all_upper = [data[m]["mean"] + data[m]["std"] for m in METHOD_ORDER]
        y_max = float(np.max(np.concatenate(all_upper))) * 1.04
        y_ticks = np.linspace(0, y_max, 6)
    scale = 2
    W, H = 1708 * scale, 620 * scale
    img = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(img)
    title_font = load_font("bold", 38 * scale)
    label_font = load_font("regular", 27 * scale)
    tick_font = load_font("regular", 20 * scale)
    legend_font = load_font("regular", 23 * scale)
    left, top = 210 * scale, 88 * scale
    plot_w, plot_h = 1100 * scale, 442 * scale
    x0, y0, x1, y1 = left, top, left + plot_w, top + plot_h
    legend_x0, legend_y0 = x1 + 30 * scale, top - 4 * scale
    legend_w, legend_h = 350 * scale, 284 * scale
    episodes = data[METHOD_ORDER[0]]["episodes"]
    x_min, x_max = float(episodes.min()), float(episodes.max())
    def sx(v): return x0 + (float(v) - x_min) / (x_max - x_min) * (x1 - x0)
    def sy(v): return y1 - float(v) / y_max * (y1 - y0)
    for tv in y_ticks:
        yy = sy(tv)
        draw.line((x0, yy, x1, yy), fill=(215, 215, 215), width=1 * scale)
        label = fmt_tick(tv, decimals=decimals)
        bbox = draw.textbbox((0, 0), label, font=tick_font)
        draw.text((x0 - 14 * scale - (bbox[2] - bbox[0]), yy - (bbox[3] - bbox[1]) / 2), label, font=tick_font, fill=(80, 80, 80))
    for tv in [0, 200, 400, 600, 800, 1000]:
        xx = sx(tv)
        label = str(tv)
        bbox = draw.textbbox((0, 0), label, font=tick_font)
        draw.text((xx - (bbox[2] - bbox[0]) / 2, y1 + 20 * scale), label, font=tick_font, fill=(0, 0, 0))
    draw.line((x0, y0, x0, y1), fill=(0, 0, 0), width=2 * scale)
    draw.line((x0, y1, x1, y1), fill=(0, 0, 0), width=1 * scale)
    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    for method in METHOD_ORDER:
        d = data[method]
        c = LINE_COLORS[method]
        upper = np.clip(d["mean"] + d["std"], 0, y_max)
        lower = np.clip(d["mean"] - d["std"], 0, y_max)
        od.polygon(
            [(sx(ep), sy(v)) for ep, v in zip(d["episodes"], upper)] + [(sx(ep), sy(v)) for ep, v in zip(d["episodes"][::-1], lower[::-1])],
            fill=(c[0], c[1], c[2], FILL_ALPHA[method]),
        )
    img_rgba = img.convert("RGBA")
    img_rgba.alpha_composite(overlay)
    img = img_rgba.convert("RGB")
    draw = ImageDraw.Draw(img)
    for method in METHOD_ORDER:
        d = data[method]
        c = LINE_COLORS[method]
        pts = [(sx(ep), sy(v)) for ep, v in zip(d["episodes"], np.clip(d["mean"], 0, y_max))]
        draw.line(pts, fill=c, width=(3 if method == "MIRA" else 2) * scale, joint="curve")
    draw.text((x0, 24 * scale), title, font=title_font, fill=(0, 0, 0))
    draw_centered(draw, ((x0 + x1) / 2, H - 25 * scale), "episode", label_font, (0, 0, 0))
    tmp = Image.new("RGBA", (560 * scale, 60 * scale), (0, 0, 0, 0))
    td = ImageDraw.Draw(tmp)
    bbox = td.textbbox((0, 0), ylabel, font=label_font)
    td.text(((tmp.width - (bbox[2] - bbox[0])) / 2, (tmp.height - (bbox[3] - bbox[1])) / 2), ylabel, font=label_font, fill=(0, 0, 0))
    rot = tmp.rotate(90, expand=True)
    img.paste(rot.convert("RGB"), (46 * scale, int((y0 + y1 - rot.height) / 2)), rot)
    draw.rectangle((legend_x0, legend_y0, legend_x0 + legend_w, legend_y0 + legend_h), fill=(255, 255, 255), outline=(215, 215, 215), width=1 * scale)
    ly = legend_y0 + 25 * scale
    for method in METHOD_ORDER:
        c = LINE_COLORS[method]
        draw.line((legend_x0 + 13 * scale, ly + 12 * scale, legend_x0 + 50 * scale, ly + 12 * scale), fill=c, width=(3 if method == "MIRA" else 2) * scale)
        draw.text((legend_x0 + 65 * scale, ly - 1 * scale), method, font=legend_font, fill=(0, 0, 0))
        ly += 33 * scale
    img.resize((1708, 620), Image.Resampling.LANCZOS).save(FINAL_DIR / filename, quality=95)

def normalize_operator(op):
    op = (op or "").strip()
    for family in OPERATOR_ORDER:
        if op.startswith(family):
            return family
    return None

def read_operator_counts(path):
    counts = defaultdict(lambda: {op: 0 for op in OPERATOR_ORDER})
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            op = normalize_operator(row.get("operator", ""))
            if op:
                counts[int(float(row["episode"]))][op] += 1
    return {ep: dict(vals) for ep, vals in counts.items()}

def compute_operator_stats():
    result = {}
    for method, seed_paths in TRACE_FILES.items():
        seed_counts = {seed: read_operator_counts(path) for seed, path in seed_paths.items()}
        common = sorted(set.intersection(*(set(c.keys()) for c in seed_counts.values())))
        counts = {}
        for op in OPERATOR_ORDER:
            arr = []
            for seed in sorted(seed_counts):
                arr.append(moving_average([seed_counts[seed][ep].get(op, 0) for ep in common]))
            counts[op] = np.vstack(arr).mean(axis=0)
        result[method] = {"episodes": np.asarray(common, dtype=float), "counts": counts}
    return result

def plot_operator_distribution():
    data = compute_operator_stats()
    scale = 2
    W, H = 1708 * scale, 1450 * scale
    img = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(img)
    title_font = load_font("bold", 38 * scale)
    panel_font = load_font("bold", 24 * scale)
    tick_font = load_font("regular", 18 * scale)
    legend_font = load_font("regular", 22 * scale)
    draw.text((116 * scale, 35 * scale), "Covertype Operator Distribution Evolution", font=title_font, fill=(0, 0, 0))
    plot_w, plot_h = 600 * scale, 265 * scale
    x_gap, y_gap = 76 * scale, 70 * scale
    left0, top0 = 116 * scale, 112 * scale
    positions = [(left0, top0), (left0 + plot_w + x_gap, top0), (left0, top0 + plot_h + y_gap), (left0 + plot_w + x_gap, top0 + plot_h + y_gap), (left0, top0 + 2 * (plot_h + y_gap)), (left0 + plot_w + x_gap, top0 + 2 * (plot_h + y_gap)), (left0, top0 + 3 * (plot_h + y_gap))]
    for method, (x0, y0) in zip(METHOD_ORDER, positions):
        x1, y1 = x0 + plot_w, y0 + plot_h
        draw.text((x0, y0 - 32 * scale), method, font=panel_font, fill=(0, 0, 0))
        eps = data[method]["episodes"]
        def sx(v): return x0 + (float(v) - float(eps.min())) / (float(eps.max()) - float(eps.min())) * (x1 - x0)
        def sy(v): return y1 - float(v) / 500.0 * (y1 - y0)
        for tv in [0, 100, 200, 300, 400, 500]:
            yy = sy(tv)
            draw.line((x0, yy, x1, yy), fill=(222, 222, 222), width=1 * scale)
            label = str(tv)
            bbox = draw.textbbox((0, 0), label, font=tick_font)
            draw.text((x0 - 10 * scale - (bbox[2] - bbox[0]), yy - (bbox[3] - bbox[1]) / 2), label, font=tick_font, fill=(90, 90, 90))
        for tv in [0, 250, 500, 750, 1000]:
            xx = sx(tv)
            label = str(tv)
            bbox = draw.textbbox((0, 0), label, font=tick_font)
            draw.text((xx - (bbox[2] - bbox[0]) / 2, y1 + 14 * scale), label, font=tick_font, fill=(90, 90, 90))
        baseline = np.zeros_like(eps)
        overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        od = ImageDraw.Draw(overlay)
        for op in OPERATOR_ORDER:
            vals = data[method]["counts"][op]
            top = baseline + vals
            c = OPERATOR_COLORS[op]
            od.polygon([(sx(ep), sy(v)) for ep, v in zip(eps, top)] + [(sx(ep), sy(v)) for ep, v in zip(eps[::-1], baseline[::-1])], fill=(c[0], c[1], c[2], 255))
            baseline = top
        img_rgba = img.convert("RGBA")
        img_rgba.alpha_composite(overlay)
        img = img_rgba.convert("RGB")
        draw = ImageDraw.Draw(img)
        draw.line((x0, y0, x0, y1), fill=(0, 0, 0), width=2 * scale)
        draw.line((x0, y1, x1, y1), fill=(0, 0, 0), width=1 * scale)
    legend_x, legend_y = 1415 * scale, 108 * scale
    draw.rectangle((legend_x, legend_y, legend_x + 268 * scale, legend_y + 180 * scale), fill=(255, 255, 255), outline=(215, 215, 215), width=1 * scale)
    ly = legend_y + 22 * scale
    for op in OPERATOR_ORDER:
        c = OPERATOR_COLORS[op]
        draw.rectangle((legend_x + 14 * scale, ly + 3 * scale, legend_x + 42 * scale, ly + 22 * scale), fill=c)
        draw.text((legend_x + 58 * scale, ly - 3 * scale), op, font=legend_font, fill=(0, 0, 0))
        ly += 34 * scale
    img.resize((1708, 1450), Image.Resampling.LANCZOS).save(FINAL_DIR / "covertype_operator_distribution_evolution.png", quality=95)

def main():
    plot_line_metric("extrinsic_reward", "Covertype Extrinsic Reward", "Extrinsic Reward", "covertype_extrinsic_reward.png")
    plot_line_metric("cumulative_extrinsic_reward", "Covertype Cumulative Extrinsic Reward", "Cumulative Extrinsic Reward", "covertype_cumulative_extrinsic_reward.png", y_max=60000, y_ticks=[0, 20000, 40000, 60000])
    plot_line_metric("target_efficiency", "Covertype Target Efficiency", "Target Efficiency", "covertype_target_efficiency.png", y_max=2.0, y_ticks=[0, 0.5, 1.0, 1.5, 2.0], decimals=True)
    plot_line_metric("cumulative_target_efficiency", "Covertype Cumulative Target Efficiency", "Cumulative Target Efficiency", "covertype_cumulative_target_efficiency.png", y_max=1.2, y_ticks=[0, 0.3, 0.6, 0.9, 1.2], decimals=True)
    plot_line_metric("cumulative_unique_sets_viewed", "Covertype Cumulative Unique Sets Viewed", "Cumulative Unique Sets Viewed", "covertype_cumulative_unique_sets_viewed.png", y_max=80000, y_ticks=[0, 20000, 40000, 60000, 80000])
    plot_operator_distribution()
    print(f"Wrote final figures to {FINAL_DIR}")

if __name__ == "__main__":
    main()
