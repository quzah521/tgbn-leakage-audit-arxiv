"""Figure 4: test NDCG@10 of single channels without leakage, by memory (Section 6.1).

Each channel is placed at the mean age of the information it averages; one panel per daily
dataset.

    labels (circles)  moving averages of labels EMA<b>; mean age b / (1 - b) label intervals
    edges (squares)   decayed edge weights with time constant tau; mean age tau
    dotted line       total edge weight over the whole history (CUM)

Usage:
    python scripts/plot_memory.py                     # reads reference_results/
    python scripts/plot_memory.py --results results --out results/figures/memory.pdf
"""
import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
import make_tables as mt  # noqa: E402

DATASETS = ["tgbn-genre", "tgbn-reddit", "tgbn-token"]
C_LAB, C_EDGE = "#2a78d6", "#eb6834"
C_INK, C_GRID = "#2b2b2b", "#e4e6ea"
DEC_TAUS = [("DEC0.125P", 0.125), ("DEC0.5P", 0.5), ("DEC2P", 2.0), ("DEC8P", 8.0)]


def ema_points(test_mean):
    """(mean age, score) for every EMA channel in the summary, sorted by age."""
    pts = []
    for k, v in test_mean.items():
        if k.startswith("EMA"):
            b = float(k[3:])
            pts.append((b / (1 - b), v))
    return sorted(pts)


def collect(hdir):
    """{dataset: {"labels": [(age, score)], "edges": [(age, score)], "cum": score}}."""
    out = {}
    for name in DATASETS:
        cfg = mt.SETUP[name]
        js = mt.heur(hdir, name, cfg["lf"][0], cfg["weight"], cfg["lf"][1])
        if js is None:
            continue
        tmr = js["test_mean"]["real"]
        out[name] = {"labels": ema_points(tmr),
                     "edges": [(tau, tmr[k]) for k, tau in DEC_TAUS if k in tmr],
                     "cum": tmr.get("CUM")}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=os.path.join(HERE, "..", "reference_results"))
    ap.add_argument("--out", default="memory.pdf")
    ap.add_argument("--png", default=None)
    ap.add_argument("--width", type=float, default=3.35, help="inches (one PVLDB column ~3.33 in)")
    ap.add_argument("--height", type=float, default=1.55)
    a = ap.parse_args(argv)
    data = collect(os.path.join(a.results, "heuristics"))
    for n, d in data.items():
        print(n, "labels", [(round(x, 2), round(y, 4)) for x, y in d["labels"]],
              "edges", [(x, round(y, 4)) for x, y in d["edges"]], "CUM", d["cum"])
    if mt.missing:
        print("missing:", *mt.missing, sep="\n  ")

    plt.rcParams.update({
        "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7, "xtick.labelsize": 6, "ytick.labelsize": 6,
        "legend.fontsize": 6.5, "pdf.fonttype": 42, "ps.fonttype": 42, "axes.edgecolor": "#9a9ea6",
        "axes.linewidth": 0.6, "xtick.color": C_INK, "ytick.color": C_INK, "text.color": C_INK,
        "axes.labelcolor": C_INK,
    })
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(a.width, a.height))
    for ax, name in zip(axes, DATASETS):
        d = data.get(name)
        ax.set_title(name.split("-")[1], pad=3)
        ax.set_xscale("log")
        ax.set_xlim(0.08, 30)
        ax.set_xticks([0.1, 1, 10])
        ax.set_xticklabels(["0.1", "1", "10"])
        ax.yaxis.grid(True, color=C_GRID, lw=0.5, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(length=2.5, width=0.6, pad=1.5)
        if d is None:
            continue
        if d["cum"] is not None:
            ax.axhline(d["cum"], color=C_EDGE, lw=0.8, ls=(0, (1, 1.5)), zorder=1)
        for key, color, marker in (("labels", C_LAB, "o"), ("edges", C_EDGE, "s")):
            xs, ys = zip(*d[key]) if d[key] else ((), ())
            ax.plot(xs, ys, color=color, lw=1.2, marker=marker, ms=3.0, zorder=3)
    axes[0].set_ylabel("NDCG@10", labelpad=2)
    axes[len(axes) // 2].set_xlabel("mean age of the averaged information (days)", labelpad=2)
    handles = [Line2D([], [], color=C_LAB, marker="o", ms=3.0, lw=1.2, label="labels (EMA)"),
               Line2D([], [], color=C_EDGE, marker="s", ms=3.0, lw=1.2, label="edges (decayed)"),
               Line2D([], [], color=C_EDGE, lw=0.8, ls=(0, (1, 1.5)), label="edges (all history)")]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.53, 0.99),
               handlelength=1.8, columnspacing=0.9, handletextpad=0.4)
    fig.tight_layout(rect=(0, 0, 1, 0.9), w_pad=0.6)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    fig.savefig(a.out, bbox_inches="tight", pad_inches=0.02)
    if a.png:
        fig.savefig(a.png, dpi=300, bbox_inches="tight", pad_inches=0.02)
    print("written", a.out, a.png or "")


if __name__ == "__main__":
    main()
