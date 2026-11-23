"""Figure 2: the label-window probe on the four TGB node affinity datasets.

One panel per dataset. Bars: coverage c(a). Lines: mean cosine similarity s(a) under unit
weights (solid) and under absolute edge features (dashed). Gray band: documented window.
Orange bar: label window recovered by the probe.

Usage:
    python scripts/plot_probe.py                      # reads reference_results/probe/
    python scripts/plot_probe.py --results results/probe --out results/figures/probe.pdf \
        --png results/figures/probe.png
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATASETS = ["tgbn-trade", "tgbn-genre", "tgbn-reddit", "tgbn-token"]  # order of Table 1

# colors; the two similarity lines also differ in dash and marker, so they stay apart in grayscale
C_UNIT = "#2a78d6"      # similarity, unit weights
C_ABS = "#16956a"       # similarity, absolute edge feature
C_COV = "#cfd3d8"       # coverage bars
C_DOC = "#e4e9f0"       # documented window
C_REC = "#c55a11"       # recovered window (the leak color of Figure 1)
C_INK = "#2b2b2b"
C_GRID = "#e4e6ea"


def load(results: str, name: str) -> dict:
    """Read <results>/<name>.json written by `python -m label_probe tgb`."""
    with open(os.path.join(results, f"{name}.json")) as f:
        js = json.load(f)
    rows = sorted(js["rows"], key=lambda r: r["offset"])
    summ = js["summary"]
    return {
        "a": [r["offset"] for r in rows],
        "cov": [r["row_coverage"] for r in rows],
        "unit": [r["cos_count"] for r in rows],
        "abs": [r["cos_abs"] for r in rows],
        "window": tuple(summ["window"]),
        "documented": tuple(summ["documented"]),
        "unit_label": js.get("unit_label", "day"),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=os.path.join(HERE, "..", "reference_results", "probe"),
                    help="directory with the <name>.json files of python -m label_probe")
    ap.add_argument("--out", default="probe.pdf")
    ap.add_argument("--png", default=None, help="optional PNG preview")
    ap.add_argument("--xmin", type=int, default=-3)
    ap.add_argument("--xmax", type=int, default=8)
    ap.add_argument("--width", type=float, default=7.0, help="inches (PVLDB text width ~7 in)")
    ap.add_argument("--height", type=float, default=1.85)
    a = ap.parse_args(argv)

    plt.rcParams.update({
        "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7, "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5, "legend.fontsize": 7, "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.edgecolor": "#9a9ea6", "axes.linewidth": 0.6, "xtick.color": C_INK, "ytick.color": C_INK,
        "text.color": C_INK, "axes.labelcolor": C_INK,
    })
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(a.width, a.height), sharey=True)
    for ax, name in zip(axes, DATASETS):
        d = load(a.results, name)
        keep = [i for i, x in enumerate(d["a"]) if a.xmin <= x <= a.xmax]
        xs = [d["a"][i] for i in keep]
        d0, d1 = d["documented"]
        ax.axvspan(d0 - 0.5, d1 + 0.5, color=C_DOC, zorder=0, lw=0)
        ax.bar(xs, [d["cov"][i] for i in keep], width=0.62, color=C_COV, zorder=1, lw=0)
        ax.plot(xs, [d["unit"][i] for i in keep], color=C_UNIT, lw=1.2, marker="o", ms=2.8, zorder=3)
        ax.plot(xs, [d["abs"][i] for i in keep], color=C_ABS, lw=1.2, ls=(0, (3, 1.6)), marker="s",
                ms=2.6, zorder=3)
        w0, w1 = d["window"]
        ax.plot([w0 - 0.42, w1 + 0.42], [1.075, 1.075], color=C_REC, lw=3.2, solid_capstyle="butt",
                zorder=4, clip_on=False)
        ax.set_xlim(a.xmin - 0.6, a.xmax + 0.6)
        ax.set_ylim(0, 1.12)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xticks(range(a.xmin + (a.xmin % 2), a.xmax + 1, 2))
        ax.set_xticks(range(a.xmin, a.xmax + 1), minor=True)
        ax.yaxis.grid(True, color=C_GRID, lw=0.5, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(length=2.5, width=0.6, which="major")
        ax.tick_params(length=1.5, width=0.5, which="minor")
        unit = "years" if d["unit_label"] == "year" else "days"
        ax.set_xlabel(f"offset $a$ ({unit})", labelpad=1.5)
        ax.set_title(name, pad=9)
    axes[0].set_ylabel("similarity / coverage", labelpad=2)
    handles = [
        Line2D([], [], color=C_UNIT, lw=1.2, marker="o", ms=2.8, label="similarity, unit weights"),
        Line2D([], [], color=C_ABS, lw=1.2, ls=(0, (3, 1.6)), marker="s", ms=2.6,
               label="similarity, absolute feature"),
        Patch(color=C_COV, label="coverage"),
        Patch(color=C_DOC, label="documented window"),
        Line2D([], [], color=C_REC, lw=3.2, label="recovered window"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.02),
               handlelength=2.2, columnspacing=1.4)
    fig.tight_layout(rect=(0, 0, 1, 0.88), w_pad=0.6)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    fig.savefig(a.out, bbox_inches="tight", pad_inches=0.02)
    if a.png:
        fig.savefig(a.png, dpi=300, bbox_inches="tight", pad_inches=0.02)
    print("written", a.out, a.png or "")


if __name__ == "__main__":
    main()
