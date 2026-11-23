"""Figure 3: label-based minus edge-based test NDCG@10 under both protocols (Section 5.3).

Left: label blend minus edge blend, with the 95% bootstrap CI over test label times (not drawn
on tgbn-trade, which has 3 test label times). Right: NAVIS label mode minus edge mode, from the
means over seeds; hollow markers involve a value reported in the NAVIS paper.

Usage:
    python scripts/plot_inputs.py                     # reads reference_results/
    python scripts/plot_inputs.py --results results --out results/figures/inputs.pdf
"""
import argparse
import json
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

DATASETS = ["tgbn-trade", "tgbn-genre", "tgbn-reddit", "tgbn-token"]
# one color and one marker per dataset, so that the series stay apart in grayscale
STYLE = {"tgbn-trade": ("#2a78d6", "o"), "tgbn-genre": ("#eb6834", "s"),
         "tgbn-reddit": ("#1baf7a", "D"), "tgbn-token": ("#eda100", "^")}
C_INK, C_GRID, C_ZERO = "#2b2b2b", "#e4e6ea", "#9a9ea6"


def heuristic_diffs(hdir):
    """{dataset: [(label - edge blend, 95% CI, from paper?) for Orig. and LF]}."""
    out = {}
    for name in DATASETS:
        cfg = mt.SETUP[name]
        pts = []
        for proto in ("orig", "lf"):
            js = mt.heur(hdir, name, cfg[proto][0], cfg["weight"], cfg[proto][1])
            if js is None:
                pts.append(None)
                continue
            c = js["comparisons"].get(mt.CMP_LE)
            d = mt.tm(js, "Fit-LabelOnly") - mt.tm(js, "Fit-EdgeOnly")
            ci = None if (c is None or name == "tgbn-trade") else c["ci95"]
            pts.append((d, ci, False))
        out[name] = pts
    return out


def navis_diffs(recs, paper):
    """{dataset: [(label - edge mode of NAVIS, None, from paper?) for Orig. and LF]}."""
    out = {}
    for name in DATASETS:
        nv = mt.navis_cells(recs, paper, name)
        pts = []
        for i in (0, 1):
            lab, edg = nv["label"][i], nv["edge"][i]
            if lab[0] is None or edg[0] is None:
                pts.append(None)
            else:
                pts.append((lab[0]["mean"] - edg[0]["mean"], None, lab[1] or edg[1]))
        out[name] = pts
    return out


def draw(ax, diffs, title):
    ax.axhline(0, color=C_ZERO, lw=0.7, ls=(0, (3, 2)), zorder=1)
    for k, name in enumerate(DATASETS):
        color, marker = STYLE[name]
        pts = diffs[name]
        xs = [x + (k - 1.5) * 0.035 for x in (0, 1)]  # small horizontal dodge
        ok = [(x, p) for x, p in zip(xs, pts) if p is not None]
        if len(ok) == 2:
            ax.plot([x for x, _ in ok], [p[0] for _, p in ok], color=color, lw=1.2, zorder=2)
        for x, p in ok:
            d, ci, paper = p
            if ci is not None:
                ax.plot([x, x], ci, color=color, lw=1.0, zorder=3, solid_capstyle="butt")
            ax.plot([x], [d], marker=marker, ms=3.6, color=color, zorder=4, ls="none",
                    markerfacecolor="white" if paper else color, markeredgewidth=0.9)
    ax.set_xlim(-0.35, 1.35)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Orig.", "LF"])
    ax.yaxis.grid(True, color=C_GRID, lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=2.5, width=0.6)
    ax.set_title(title, pad=4)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=os.path.join(HERE, "..", "reference_results"))
    ap.add_argument("--paper", default=os.path.join(HERE, "..", "reference_results", "navis",
                                                    "navis_paper_reported.json"))
    ap.add_argument("--out", default="inputs.pdf")
    ap.add_argument("--png", default=None)
    ap.add_argument("--width", type=float, default=3.35, help="inches (one PVLDB column ~3.33 in)")
    ap.add_argument("--height", type=float, default=1.75)
    a = ap.parse_args(argv)
    hd = heuristic_diffs(os.path.join(a.results, "heuristics"))
    nd = navis_diffs(mt.load_navis(os.path.join(a.results, "navis", "navis_audit_results.jsonl")),
                     json.load(open(a.paper)))
    for name in DATASETS:
        print(name, "heuristics", [None if p is None else round(p[0], 4) for p in hd[name]],
              "NAVIS", [None if p is None else round(p[0], 4) for p in nd[name]])

    plt.rcParams.update({
        "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5, "legend.fontsize": 6.5, "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.edgecolor": "#9a9ea6", "axes.linewidth": 0.6, "xtick.color": C_INK, "ytick.color": C_INK,
        "text.color": C_INK, "axes.labelcolor": C_INK,
    })
    fig, axes = plt.subplots(1, 2, figsize=(a.width, a.height), sharey=True)
    draw(axes[0], hd, "Training-free blends")
    draw(axes[1], nd, "NAVIS (label vs. edge mode)")
    axes[0].set_ylabel("label $-$ edge NDCG@10", labelpad=2)
    handles = [Line2D([], [], color=STYLE[n][0], marker=STYLE[n][1], ms=3.6, lw=1.2, label=n.split("-")[1])
               for n in DATASETS]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.53, 1.03),
               handlelength=1.8, columnspacing=1.0, handletextpad=0.4)
    fig.tight_layout(rect=(0, 0, 1, 0.88), w_pad=0.8)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    fig.savefig(a.out, bbox_inches="tight", pad_inches=0.02)
    if a.png:
        fig.savefig(a.png, dpi=300, bbox_inches="tight", pad_inches=0.02)
    print("written", a.out, a.png or "")


if __name__ == "__main__":
    main()
