"""Collect run outputs into results/summary.json, a markdown table and the two README figures.

Usage (after copying run outputs into outputs/): python act_onnx/report.py
Needs numpy + matplotlib only.
"""

import glob
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt

OUT = Path("outputs")
RESULTS, DOCS = Path("results"), Path("docs")
# Reference categorical palette, first three slots (validated all-pairs in light mode).
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

LABELS = {
    "pytorch_fp32": "PyTorch FP32",
    "onnx_fp32": "ONNX FP32",
    "onnx_int8_minmax": "INT8 (MinMax)",
    "onnx_int8_percentile": "INT8 (Percentile)",
    **{f"onnx_mixed_top{k}": f"Mixed: top-{k} FP32" for k in (1, 2, 3, 5, 8)},
    "pytorch_noise_0p001": "PyTorch + noise σ=0.001",
    "pytorch_noise_0p032": "PyTorch + noise σ=0.032",
}


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


def load():
    runs = {}
    for f in glob.glob(str(OUT / "closed_loop/*/eval_info.json")):
        name = Path(f).parent.name.split("_seed")[0]
        runs[name] = json.load(open(f))["per_task"][0]["metrics"]["successes"]
    ref = runs["pytorch_fp32"]
    sens = json.load(open(OUT / "sensitivity/results.json"))
    offline = {  # mean |error| of the normalized action chunk vs PyTorch, held-out frames
        "onnx_fp32": json.load(open(OUT / "onnx/parity_fp32.json"))["normalized"]["mean_abs"],
        "onnx_int8_percentile": json.load(open(OUT / "onnx/parity_int8_percentile.json"))["normalized"]["mean_abs"],
        "onnx_int8_minmax": sens["full_int8"]["normalized"]["mean_abs"],
        **{f"onnx_mixed_top{k}": v["normalized"]["mean_abs"] for k, v in sens["mixed"].items()},
        # E|N(0, σ²)| = σ·sqrt(2/π)
        "pytorch_noise_0p001": 0.001 * math.sqrt(2 / math.pi),
        "pytorch_noise_0p032": 0.032 * math.sqrt(2 / math.pi),
    }
    lat = {}
    for f in glob.glob(str(OUT / "latency/laptop_*t.json")):
        d = json.load(open(f))
        for m, v in d["models"].items():
            lat.setdefault(m.replace("act_", "onnx_"), {})[f"{d['threads']}t"] = v
    rows = {}
    for name, s in runs.items():
        k, n = sum(s), len(s)
        gained = sum((not a) and b for a, b in zip(ref, s))
        lost = sum(a and not b for a, b in zip(ref, s))
        rows[name] = {
            "label": LABELS.get(name, name),
            "success": k, "episodes": n, "ci95": wilson(k, n),
            "flips_vs_pytorch": gained + lost, "gained": gained, "lost": lost,
            "offline_mean_abs_norm": offline.get(name),
            "latency": lat.get(name, {}),
        }
    return rows, sens


def table(rows):
    order = ["pytorch_fp32", "onnx_fp32", "onnx_int8_minmax", "onnx_int8_percentile",
             *[f"onnx_mixed_top{k}" for k in (1, 2, 3, 5, 8)], "pytorch_noise_0p001", "pytorch_noise_0p032"]
    lines = [
        "| Variant | Size (MB) | Laptop latency, 4 thr / 1 thr (ms) | Offline mean abs error | Success /100 (95% CI) | Outcomes flipped vs PyTorch |",
        "|---|---|---|---|---|---|",
    ]
    for name in order:
        r = rows[name]
        lat = r["latency"]
        size = f"{lat['4t']['size_mb']:.0f}" if lat else "—"
        ms = f"{lat['4t']['median_ms']:.0f} / {lat['1t']['median_ms']:.0f}" if lat else "—"
        err = "0 (reference)" if name == "pytorch_fp32" else f"{r['offline_mean_abs_norm']:.1e}"
        lo, hi = r["ci95"]
        flips = "—" if name == "pytorch_fp32" else f"{r['flips_vs_pytorch']} (+{r['gained']} / −{r['lost']})"
        lines.append(f"| {r['label']} | {size} | {ms} | {err} | **{r['success']}** ({lo:.0f}–{hi:.0f}) | {flips} |")
    return "\n".join(lines)


def style(ax, title, xlabel, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=INK, fontsize=12, pad=12)
    ax.set_xlabel(xlabel, color=INK2)
    ax.set_ylabel(ylabel, color=INK2)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2)


def fig_success_latency(rows):
    fig, ax = plt.subplots(figsize=(8, 4.8), facecolor=SURFACE)
    ref = rows["pytorch_fp32"]
    lo, hi = ref["ci95"]
    ax.axhspan(lo, hi, color=GRID, alpha=0.6, linewidth=0)
    ax.axhline(ref["success"], color=INK2, linewidth=1, linestyle="--")
    ax.text(335, lo - 2.2, "dashed: PyTorch FP32 84%, shaded: its 95% CI", color=INK2, fontsize=9, ha="right")
    offsets = {"onnx_fp32": (-10, -3, "right"), "onnx_int8_minmax": (-10, 4, "right"),
               "onnx_mixed_top1": (-10, -10, "right"), "onnx_mixed_top2": (8, -10, "left")}
    groups = [("onnx_fp32", BLUE, "ONNX FP32"), ("onnx_int8_minmax", ORANGE, "INT8"),
              *[(f"onnx_mixed_top{k}", AQUA, "Mixed precision") for k in (1, 2, 3, 5, 8)]]
    seen = set()
    for name, color, legend in groups:
        r = rows[name]
        x = r["latency"]["1t"]["median_ms"]
        lo, hi = r["ci95"]
        ax.errorbar(x, r["success"], yerr=[[r["success"] - lo], [hi - r["success"]]], fmt="o", ms=8,
                    color=color, ecolor=color, elinewidth=1.5, capsize=0, markeredgecolor=SURFACE,
                    markeredgewidth=2, label=None if legend in seen else legend, zorder=3)
        seen.add(legend)
        short = r["label"].replace("Mixed: ", "").replace(" FP32", "") if "mixed" in name else r["label"]
        dx, dy, ha = offsets.get(name, (8, -3, "left"))
        ax.annotate(short, (x, r["success"]), xytext=(dx, dy), textcoords="offset points", fontsize=8.5,
                    color=INK, ha=ha)
    ax.set_xlim(60, 340)
    ax.set_ylim(60, 100)
    style(ax, "Closed-loop success vs CPU latency (100 episodes each)",
          "Latency per action chunk, laptop CPU, 1 thread (ms, median)", "Task success (%)")
    ax.legend(frameon=False, loc="lower right", labelcolor=INK)
    fig.tight_layout()
    fig.savefig(DOCS / "success_vs_latency.png", dpi=160, facecolor=SURFACE)


def fig_flips(rows):
    fig, ax = plt.subplots(figsize=(8, 4.8), facecolor=SURFACE)
    groups = [(["onnx_fp32"], BLUE, "ONNX FP32"),
              (["onnx_int8_minmax", "onnx_int8_percentile", *[f"onnx_mixed_top{k}" for k in (1, 2, 3, 5, 8)]],
               ORANGE, "INT8 / mixed precision"),
              (["pytorch_noise_0p001", "pytorch_noise_0p032"], AQUA, "Control: PyTorch + Gaussian noise")]
    for names, color, legend in groups:
        xs = [rows[n]["offline_mean_abs_norm"] for n in names]
        ys = [rows[n]["flips_vs_pytorch"] for n in names]
        ax.scatter(xs, ys, s=64, color=color, edgecolors=SURFACE, linewidths=2, label=legend, zorder=3)
    for n in ("onnx_fp32", "onnx_int8_minmax", "onnx_mixed_top8", "pytorch_noise_0p001", "pytorch_noise_0p032"):
        r = rows[n]
        ax.annotate(r["label"], (r["offline_mean_abs_norm"], r["flips_vs_pytorch"]), xytext=(8, 4),
                    textcoords="offset points", fontsize=8.5, color=INK)
    ax.set_xscale("log")
    ax.set_xlim(1e-7, 0.2)
    ax.set_ylim(0, 22)
    style(ax, "Episodes whose outcome flips vs PyTorch, by offline error",
          "Offline mean |error| of the normalized action chunk vs PyTorch (log)",
          "Outcomes flipped (of 100 episodes)")
    ax.legend(frameon=False, loc="upper left", labelcolor=INK)
    fig.tight_layout()
    fig.savefig(DOCS / "flips_vs_offline_error.png", dpi=160, facecolor=SURFACE)


def main():
    RESULTS.mkdir(exist_ok=True)
    DOCS.mkdir(exist_ok=True)
    rows, sens = load()
    summary = {"closed_loop": rows, "sensitivity": {k: sens[k] for k in ("method", "n_eval", "groups", "ranking")},
               "sensitivity_only_group_action_units": {g: v["action_units"] for g, v in sens["only_group"].items()}}
    with open(RESULTS / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    md = table(rows)
    (RESULTS / "table.md").write_text(md + "\n")
    print(md)
    fig_success_latency(rows)
    fig_flips(rows)


if __name__ == "__main__":
    main()
