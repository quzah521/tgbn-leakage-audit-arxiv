"""Collect all outputs into the numbers of the paper's tables (Markdown, optionally LaTeX).

In this partial release, the heuristic and NAVIS outputs come from reference_results/; the
pipeline that produces them will be released on publication (reproduce.py tables does this).

Usage:
    python scripts/make_tables.py                                # reads results/
    python scripts/make_tables.py --results reference_results    # our reference outputs
    python scripts/make_tables.py --latex results/tables.tex     # also LaTeX rows (Tables 4, 5; S1, S3)

Expected layout under --results (each part can be overridden):
  probe/<name>.json                    python -m label_probe tgb --all --out_dir results/probe
  loop/loop_diagnostics.json           python scripts/audit_loop.py
  heuristics/<run>/summary.json        re-evaluation pipeline (reference outputs in this release)
  navis/navis_audit_results.jsonl      NAVIS runs (reference outputs in this release)

Tables (numbered as in the paper; S = supplementary tables whose numbers the text quotes)
  Table 3   label windows, availability delays, loop diagnostics (test split)
  S1        loop diagnostics on all splits (Section 4.3)
  S2        information revealed by each form of leakage, parameter-free methods (Section 4.4)
  Table 4   test NDCG@10 under the original and the leakage-free protocol (mean ± std over seeds)
  Table 5   ranks under both protocols and Kendall's tau
  S3        label-release delay sweep on tgbn-reddit (Section 5.5)
  S4        label-based minus edge-based, the numbers behind Figure 3 (Section 5.3)
  S5        edge weighting on tgbn-reddit (Section 5.5)
  S6        within-period placebo (Section 6.3)
  S7        all NAVIS runs, one row per configuration
Every missing input is listed at the end with the command that produces it.
"""
import argparse
import json
import os
import sys
from collections import defaultdict
from decimal import ROUND_HALF_UP, Decimal

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from tgbn_audit.data import AVAILABILITY_DELAY, TGB_MA_WINDOW  # noqa: E402

DATASETS = ["tgbn-trade", "tgbn-genre", "tgbn-reddit", "tgbn-token"]
# Section 5.1: edge weight of the heuristics, and the (cutoff, label delay) of each protocol.
# On the daily datasets both protocols use the strict cutoff, so that they differ only in label release.
SETUP = {
    "tgbn-trade": {"weight": "abs", "orig": ("protocol", 0), "lf": ("strict", 0)},
    "tgbn-genre": {"weight": "abs", "orig": ("strict", 0), "lf": ("strict", 7)},
    "tgbn-reddit": {"weight": "one", "orig": ("strict", 0), "lf": ("strict", 6)},
    "tgbn-token": {"weight": "abs", "orig": ("strict", 0), "lf": ("strict", 6)},
}
HEUR_ROWS = [("PF", "PF"), ("MA", "MA"), ("EMA0.95", "EMA ($\\beta=0.95$)"),
             ("Fit-LabelOnly", "Label blend"), ("Fit-EdgeOnly", "Edge blend"), ("Fit-Time", "Hybrid blend")]
CMP_LE = "real: Fit-LabelOnly vs Fit-EdgeOnly"
CMP_DID = "timing gain: real minus placebo (difference-in-differences)"
RECENCY = "DEC0.125P"  # recency counter of S2: tau = P/8 = 3 hours on the daily datasets

missing = []


# ---- loading ------------------------------------------------------------------------------------
def heur(hdir, name, cutoff, weight, delay=0):
    sub = f"{name}_{cutoff}_{weight}" + ("" if delay == 0 else f"_delay{delay:g}")
    path = os.path.join(hdir, sub, "summary.json")
    if not os.path.exists(path):
        dl = "" if delay == 0 else f", label delay {delay:g}"
        missing.append(f"{path}  <-  heuristic run {name}, cutoff {cutoff}, weight {weight}{dl} "
                       "(use --heuristics reference_results/heuristics)")
        return None
    with open(path) as f:
        return json.load(f)


def tm(js, key, run="real"):
    """Test mean of one configuration; "MA" is the MovingAverage of TGB (EMA with b = (k-1)/k)."""
    if js is None:
        return None
    d = js["test_mean"].get(run, {})
    if key == "MA" and "MA" not in d:  # some summaries store MA only under its EMA channel name
        k = TGB_MA_WINDOW.get(js.get("args", {}).get("name"))
        return None if k is None else d.get(f"EMA{(k - 1) / k:g}")
    return d.get(key)


def load_json(path, cmd):
    if not os.path.exists(path):
        missing.append(f"{path}  <-  {cmd}")
        return None
    with open(path) as f:
        return json.load(f)


def load_navis(path):
    """Read the NAVIS result lines, dropping exact duplicates; fill in defaults of older lines."""
    if not os.path.exists(path):
        missing.append(f"{path}  <-  NAVIS runs (use --navis reference_results/navis/navis_audit_results.jsonl)")
        return []
    recs, seen = [], set()
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        r.setdefault("edge_weight", "abs")
        r["edge_weight"] = r["edge_weight"] or "abs"
        r.setdefault("reg_device", "cuda")
        r["label_delay"] = float(r.get("label_delay", 0.0))
        key = json.dumps(r, sort_keys=True)
        if key not in seen:
            seen.add(key)
            recs.append(r)
    return recs


def navis_pick(recs, name, multi, cutoff=None, gs=None, delay=0.0, weight=None):
    """NAVIS runs of one configuration, one per seed (the last line of a seed wins)."""
    out = [r for r in recs if r["name"] == name and bool(r["multi"]) == multi
           and abs(r["label_delay"] - delay) < 1e-9
           and (cutoff is None or r["cutoff"] == cutoff)
           and (gs is None or r["global_state"] in gs)
           and (weight is None or r["edge_weight"] == weight)]
    return list({r["seed"]: r for r in out}.values())


def agg(runs, field="test_at_best_val"):
    """Mean, sample standard deviation and number of runs."""
    if not runs:
        return None
    v = np.array([r[field] for r in runs], dtype=float)
    return {"mean": float(v.mean()), "std": float(v.std(ddof=1)) if v.size > 1 else 0.0, "n": int(v.size)}


def navis_cells(recs, paper, name):
    """(orig, lf) for the label and the edge mode; orig falls back to the NAVIS paper (dagger)."""
    d = AVAILABILITY_DELAY[name]
    lab_o = agg(navis_pick(recs, name, False, gs=("leaky",)))
    lab_l = agg(navis_pick(recs, name, False, gs=("prev",), delay=float(d)))
    edg_o = agg(navis_pick(recs, name, True, "protocol", weight="abs"))
    edg_l = agg(navis_pick(recs, name, True, "strict", weight="abs"))
    p_lab = paper["label_setting_table2"].get(name, {}).get("test")
    p_edg = paper["full_ctdg_table1"].get(name, {}).get("test")
    wrap = lambda a, p: (a, False) if a is not None else ((None, False) if p is None else
                                                          ({"mean": p, "std": 0.0, "n": 0}, True))
    return {"label": (wrap(lab_o, p_lab), (lab_l, False)), "edge": (wrap(edg_o, p_edg), (edg_l, False))}


# ---- formatting ---------------------------------------------------------------------------------
def r3(x):
    """Round half up to three decimals (the reference summaries keep four decimals)."""
    return str(Decimal(f"{x:.6f}").quantize(Decimal("0.001"), rounding=ROUND_HALF_UP))


def f3(x):
    return "—" if x is None else r3(x)


def fdiff(x):
    return "—" if x is None else ("0.000" if abs(x) < 5e-4 else f"{x:+.3f}")


def fa(cell):
    a, paper = cell
    if a is None:
        return "—"
    s = r3(a["mean"]) + ("†" if paper else "")
    if a["n"] > 1:
        s += f" ± {a['std']:.4f} (n={a['n']})"
    return s


def mean(cell):
    return None if cell[0] is None else cell[0]["mean"]


def rel(a, b):
    return None if a is None or b is None or a == 0 else (b - a) / a * 100


def fr(x):
    return "—" if x is None else ("0.0" if abs(x) < 0.05 else f"{x:+.1f}")


def fci(c):
    return "—" if c is None else f"{c['mean_diff']:+.4f} [{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}]"


def fdelta(x, period):
    if period == 1.0:
        return f"{x:g} yr"
    for div, unit in ((86400, "d"), (3600, "h"), (60, "min")):
        if abs(x) >= div:
            return f"{x / div:.1f} {unit}"
    return f"{x:g} s"


def table(header, rows):
    return (["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
            + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


# ---- Table 3 ------------------------------------------------------------------------------------
def t3(probe_dir, loop):
    rows = []
    for name in DATASETS:
        pj = load_json(os.path.join(probe_dir, f"{name}.json"),
                       f"python -m label_probe tgb --name {name} --out_dir {probe_dir}")
        lp = loop.get(name) if loop else None
        s = pj["summary"] if pj else None
        unit = (pj.get("unit_label") or ("d" if pj["unit"] == 86400 else f"{pj['unit']:g}")) if pj else ""
        unit = {"day": "d", "year": "yr"}.get(unit, unit)
        win = f"[t+{s['window'][0]}, t+{s['window'][1] + 1}) {unit}" if s else "—"
        dly = f"{s['availability_delay_units']:g} {unit}" if s else "—"
        x = lp["protocol"]["test"] if lp else None
        P = lp["period"] if lp else None
        forms = []
        if s and x:
            if s["window"][0] == 0 and x["mean_n_eq"] > 0:
                forms.append("S")
            if s["availability_delay_units"] > 1:
                forms.append("E")
            if x["mean_n_gt"] > 0:
                forms.append("B")
        rows.append([name, win, dly,
                     "—" if not s else (f"{s['window'][1]:g} {unit}" if s["availability_delay_units"] > 1 else "none"),
                     "—" if not x else f"{x['mean_n_eq']:,.1f}", "—" if not x else f"{x['mean_n_gt']:.1f}",
                     "—" if not x else fdelta(x["max_delta"], P), ", ".join(forms) or "—"])
    return (["## Table 3: label windows and loop diagnostics (test split)", "",
             "Window and delay from the label-window probe; look-ahead under the TGB loop = a*·Δ "
             "(a* = last offset of the window); n̄_=, n̄_>: mean edges stamped at / after t_q "
             "processed before the prediction; max δ: largest edge horizon. Forms: S same-timestamp, "
             "E early label release, B batch boundary.", ""]
            + table(["dataset", "label window", "availability delay d", "look-ahead", "n̄_=", "n̄_>",
                     "max δ", "forms"], rows))


T5_DATA, T5_LABELS = {}, []


def t_ranks(latex):
    """Ranks of the Table 4 methods (scores rounded to three decimals share a rank) and Kendall's tau-b."""
    from scipy.stats import kendalltau
    rows, tex = [], []
    ranks, taus = {}, {}
    for name in DATASETS:
        keys = [k for k, _ in T5_LABELS if None not in T5_DATA[k][name]]
        for i in (0, 1):
            v = {k: round(T5_DATA[k][name][i], 3) for k in keys}
            for k in keys:
                ranks[(k, name, i)] = 1 + sum(1 for x in v.values() if x > v[k])
        if len(keys) > 1:
            taus[name] = kendalltau([round(T5_DATA[k][name][0], 3) for k in keys],
                                    [round(T5_DATA[k][name][1], 3) for k in keys]).statistic
    for k, label in T5_LABELS:
        cells = [f"{ranks[(k, n, 0)]} → {ranks[(k, n, 1)]}" if (k, n, 0) in ranks else "—" for n in DATASETS]
        rows.append([label.replace("$\\beta=0.95$", "β=0.95")] + cells)
        tex.append("\t\t" + label + " & " + " & ".join(c.replace(" → ", " $\\to$ ").replace("—", "\\tbd")
                                                       for c in cells) + " \\\\")
    rows.append(["Kendall's τ"] + [f"{taus[n]:.2f}" if n in taus else "—" for n in DATASETS])
    tex.append("\t\tKendall's $\\tau$ & " + " & ".join(f"{taus[n]:.2f}" if n in taus else "\\tbd"
                                                       for n in DATASETS) + " \\\\")
    if latex is not None:
        latex += ["% Table 5 (tab:ranks), body rows (group separators as in Table 4)", *tex, ""]
    return (["## Table 5: ranks, original → leakage-free (Section 5.4)", "",
             "Scores that agree to three decimals share a rank; Kendall's τ-b between the two rankings.", ""]
            + table(["method"] + [n.split("-")[1] for n in DATASETS], rows))


def t_splits(loop, latex):
    rows, tex = [], []
    for name in DATASETS:
        if not loop or name not in loop:
            continue
        P = loop[name]["period"]
        for i, ph in enumerate(["train", "val", "test"]):
            x = loop[name]["protocol"][ph]
            s = loop[name].get("strict", {}).get(ph)
            rows.append([name if i == 0 else "", ph, x["query_times"], f"{x['mean_n_eq']:,.1f}",
                         f"{x['mean_n_gt']:.1f}", x["max_n_gt"], f"{100 * x['share_queries_with_leak']:.1f}%",
                         fdelta(x["max_delta"], P),
                         "—" if s is None else f"{s['mean_n_eq'] + s['mean_n_gt']:.0f} / {fdelta(s['max_delta'], P)}"])
            n_eq = (f"{x['mean_n_eq']:,.0f}".replace(",", "{,}") if x["mean_n_eq"] >= 1000
                    else f"{x['mean_n_eq']:.1f}")
            qt = f"{x['query_times']:,}".replace(",", "{,}")
            tex.append(f"\t\t{name if i == 0 else ''} & {ph} & {qt} & {n_eq} & {x['mean_n_gt']:.1f} & "
                       f"{x['max_n_gt']} & {100 * x['share_queries_with_leak']:.1f}\\% & "
                       f"{fdelta(x['max_delta'], P)} \\\\")
        tex.append("\t\t\\midrule")
    if tex:
        tex = tex[:-1]
    if latex is not None and tex:
        latex += ["% S1 (loop diagnostics on all splits, artifact only), body rows", *tex, ""]
    return (["## S1: loop diagnostics on all splits (Section 4.3)", "",
             "TGB loop; last column: strict cutoff (edges at/after t_q, max δ), which must be 0 / negative.", ""]
            + table(["dataset", "split", "query times", "n̄_=", "n̄_>", "max n_>", "leaky", "max δ",
                     "strict"], rows))


# ---- S2 ---------------------------------------------------------------------------------------
def t_reveal(hdir):
    rows = []
    hp = heur(hdir, "tgbn-trade", "protocol", "abs")
    hs = heur(hdir, "tgbn-trade", "strict", "abs")
    a, b = tm(hp, "CNT"), tm(hs, "CNT")
    rows.append(["S", "tgbn-trade", "Edge counter", f3(a), f3(b), fdiff(None if None in (a, b) else a - b)])
    for name in DATASETS[1:]:
        cfg = SETUP[name]
        h0 = heur(hdir, name, cfg["orig"][0], cfg["weight"], cfg["orig"][1])
        h1 = heur(hdir, name, cfg["lf"][0], cfg["weight"], cfg["lf"][1])
        a, b = tm(h0, "PF"), tm(h1, "PF")
        rows.append(["E", name, "PF", f3(a), f3(b), fdiff(None if None in (a, b) else a - b)])
    for name in DATASETS[1:]:
        w = SETUP[name]["weight"]
        hp, hs = heur(hdir, name, "protocol", w), heur(hdir, name, "strict", w)
        a, b = tm(hp, RECENCY), tm(hs, RECENCY)
        rows.append(["B", name, "Recency counter", f3(a), f3(b), "—" if None in (a, b) else f"{a - b:+.4f}"])
    return (["## S2: information revealed by each form, test NDCG@10 (Section 4.4)", "",
             "TGB: reference loop (E: labels released immediately); Removed: only this form removed "
             "(E: delayed release; S, B: strict cutoff). Gain = TGB − Removed.", ""]
            + table(["form", "dataset", "method", "TGB", "Removed", "Gain"], rows))


# ---- Table 4 ------------------------------------------------------------------------------------
def t_reeval(hdir, recs, paper, latex):
    data = {}
    for name in DATASETS:
        cfg = SETUP[name]
        h0 = heur(hdir, name, cfg["orig"][0], cfg["weight"], cfg["orig"][1])
        h1 = heur(hdir, name, cfg["lf"][0], cfg["weight"], cfg["lf"][1])
        nv = navis_cells(recs, paper, name)
        data[name] = {key: (({"mean": tm(h0, key), "std": 0, "n": 1} if tm(h0, key) is not None else None, False),
                            ({"mean": tm(h1, key), "std": 0, "n": 1} if tm(h1, key) is not None else None, False))
                      for key, _ in HEUR_ROWS}
        data[name]["NAVIS-label"], data[name]["NAVIS-edge"] = nv["label"], nv["edge"]
    groups = [("Label-based", [("PF", "PF"), ("MA", "MA"), ("EMA0.95", "EMA ($\\beta=0.95$)"),
                               ("Fit-LabelOnly", "Label blend"), ("NAVIS-label", "NAVIS (label mode)")]),
              ("Edge-based", [("NAVIS-edge", "NAVIS (edge mode)"), ("Fit-EdgeOnly", "Edge blend")]),
              ("Hybrid", [("Fit-Time", "Hybrid blend")])]
    rows, tex = [], []
    for gi, (gname, items) in enumerate(groups):
        rows.append([f"*{gname}*"] + [""] * 12)
        if gi:
            tex.append("\t\t\\midrule")
        tex.append(f"\t\t\\multicolumn{{13}}{{@{{}}l}}{{\\emph{{{gname}}}}} \\\\")
        for key, label in items:
            r, t = [label.replace("$\\beta=0.95$", "β=0.95")], [label]
            for name in DATASETS:
                o, l_ = data[name][key]
                mo, ml = mean(o), mean(l_)
                r += [fa(o), fa(l_), fr(rel(mo, ml))]
                to = "\\tbd" if mo is None else r3(mo) + ("\\rlap{$^\\dagger$}" if o[1] else "")
                tl = "\\tbd" if ml is None else r3(ml)
                x = rel(mo, ml)
                td = "\\tbd" if x is None else ("0.0" if abs(x) < 0.05 else f"$-${-x:.1f}" if x < 0 else f"$+${x:.1f}")
                t += [to, tl, td]
            rows.append(r)
            tex.append("\t\t" + " & ".join(t) + " \\\\")
    T5_DATA.clear()
    T5_DATA.update({key: {n: (mean(data[n][key][0]), mean(data[n][key][1])) for n in DATASETS}
                    for _, items in groups for key, _ in items})
    T5_LABELS[:] = [(key, label) for _, items in groups for key, label in items]
    if latex is not None:
        latex += ["% Table 4 (tab:reeval), body rows (between \\midrule after the header and \\bottomrule)",
                  *tex, ""]
    head = ["method"] + [f"{n.split('-')[1]} {c}" for n in DATASETS for c in ("Orig.", "LF", "Δ%")]
    return (["## Table 4: test NDCG@10, original vs leakage-free protocol", "",
             "Heuristics: Section 5.1 setup (daily datasets: strict cutoff in both protocols). "
             "NAVIS: original = reference loop with the original global state; leakage-free = delayed "
             "release with the past global state (label mode) or strict cutoff (edge mode). "
             "† = value reported in the NAVIS paper; ± = std over seeds.", ""]
            + table(head, rows))


# ---- supplementary ------------------------------------------------------------------------------
def s_label_vs_edge(hdir, recs, paper):
    rows = []
    for name in DATASETS:
        cfg = SETUP[name]
        h0 = heur(hdir, name, cfg["orig"][0], cfg["weight"], cfg["orig"][1])
        h1 = heur(hdir, name, cfg["lf"][0], cfg["weight"], cfg["lf"][1])
        c0 = h0["comparisons"].get(CMP_LE) if h0 else None
        c1 = h1["comparisons"].get(CMP_LE) if h1 else None
        nv = navis_cells(recs, paper, name)
        d = lambda i: (None if None in (mean(nv["label"][i]), mean(nv["edge"][i]))
                       else mean(nv["label"][i]) - mean(nv["edge"][i]))
        rows.append([name, fci(c0), fci(c1), fdiff(d(0)), fdiff(d(1))])
    return (["## S4: label-based minus edge-based, Figure 3 (Section 5.3)", "",
             "Heuristics: label blend − edge blend, paired over test label times (95% bootstrap CI); "
             "NAVIS: label mode − edge mode.", ""]
            + table(["dataset", "heuristics, original", "heuristics, leakage-free", "NAVIS, original",
                     "NAVIS, leakage-free"], rows))


def s_delay_sweep(hdir, latex):
    """S3: one row per method, columns delay 0-1 d (identical), 3 d, 6 d, share of the decline at 3 d."""
    keys = [("PF", "PF"), ("MA", "MA"), ("EMA0.95", "EMA ($\\beta=0.95$)"), ("Fit-LabelOnly", "Label blend"),
            ("Fit-Time", "Hybrid blend")]
    hs = {dl: heur(hdir, "tgbn-reddit", "strict", "abs", dl) for dl in (0, 1, 3, 6)}
    rows, tex = [], []
    for k, label in keys:
        v = {dl: tm(hs[dl], k) for dl in hs}
        share = (None if None in (v[0], v[3], v[6]) or v[0] == v[6]
                 else (v[0] - v[3]) / (v[0] - v[6]) * 100)
        same = v[0] is not None and v[1] is not None and abs(v[0] - v[1]) < 5e-4
        rows.append([label.replace("$\\beta=0.95$", "β=0.95"), f3(v[0]), f3(v[1]), f3(v[3]), f3(v[6]),
                     "—" if share is None else f"{share:.0f}%"])
        tex.append(f"\t\t{label} & {f3(v[0])} & {f3(v[3])} & {f3(v[6])} & "
                   + ("\\tbd" if share is None else f"{share:.0f}\\%") + " \\\\"
                   + ("" if same else "  % delay 0 and 1 d differ!"))
    if latex is not None:
        latex += ["% S3 (delay sweep, artifact only), body rows: delay 0-1 d, 3 d, 6 d, share of the decline at 3 d", *tex, ""]
    grows = []
    for dl in (0, 5, 6, 7):  # tgbn-genre around its availability delay (7 days)
        h = heur(hdir, "tgbn-genre", "strict", "abs", dl)
        c = h["comparisons"].get(CMP_LE) if h else None
        grows.append([f"{dl} d"] + [f3(tm(h, k)) for k, _ in keys] + [f3(tm(h, "Fit-EdgeOnly")), fci(c)])
    return (["## S3: label-release delay sweep, tgbn-reddit (Section 5.5)", "",
             "Test NDCG@10 when a label is used only from its stamp + delay (strict cutoff, |score| weights); "
             "share = (0 d − 3 d) / (0 d − 6 d).", ""]
            + table(["method", "0 d", "1 d", "3 d", "6 d", "share at 3 d"], rows)
            + ["", "tgbn-genre around its availability delay (text of Section 5.5):", ""]
            + table(["delay"] + [lab.replace("$\\beta=0.95$", "β=0.95") for _, lab in keys]
                    + ["edge blend", "label − edge [95% CI]"], grows))


def s_weighting(hdir, recs):
    rows = []
    for w, lab in (("abs", "|score|"), ("one", "unit (edge count)")):
        h = heur(hdir, "tgbn-reddit", "strict", w, 6)
        nv = agg(navis_pick(recs, "tgbn-reddit", True, "strict", weight=w))
        rows.append([lab, f3(tm(h, "Fit-EdgeOnly")), fa((nv, False))])
    return (["## S5: edge weighting, tgbn-reddit, leakage-free (Section 5.5)", ""]
            + table(["edge weight", "edge blend", "NAVIS edge mode"], rows))


def s_placebo(hdir):
    rows = []
    runs = [("tgbn-genre", "strict", "abs", 7), ("tgbn-reddit", "strict", "one", 6),   # leakage-free (Section 6.3)
            ("tgbn-token", "strict", "abs", 6),
            ("tgbn-genre", "strict", "abs", 0), ("tgbn-genre", "protocol", "abs", 0),  # original protocol
            ("tgbn-reddit", "strict", "abs", 0)]
    for name, cutoff, w, dl in runs:
        sub = f"{name}_{cutoff}_{w}" + ("" if dl == 0 else f"_delay{dl:g}")
        path = os.path.join(hdir, sub, "summary.json")
        if not os.path.exists(path):
            continue
        js = json.load(open(path))
        c = js["comparisons"].get(CMP_DID)
        gr = js["comparisons"].get("real: Fit-Time vs Fit-NoTime")
        gp = js["comparisons"].get("placebo: Fit-Time vs Fit-NoTime")
        if c:
            rows.append([name, cutoff, w, f"{dl:g} d", fci(gr), fci(gp), fci(c),
                         "—" if c["wilcoxon_p"] != c["wilcoxon_p"] else f"{c['wilcoxon_p']:.2g}"])
    return (["## S6: within-period placebo (Section 6.3)", "",
             "Gain of the decayed edge weights (Fit-Time − Fit-NoTime) on the real and on the within-period-shuffled "
             "stream, and their difference (difference-in-differences).", ""]
            + table(["dataset", "cutoff", "weight", "label delay", "gain, real", "gain, placebo",
                     "real − placebo [95% CI]", "Wilcoxon p"], rows))


def s_navis_all(recs):
    groups = defaultdict(list)
    for r in recs:
        k = (r["name"], "edge" if r["multi"] else "label", "-" if r["multi"] else r["global_state"],
             r["cutoff"] if r["multi"] else "-", f"{r['label_delay']:g}", r["edge_weight"] if r["multi"] else "-")
        groups[k].append(r)
    rows = []
    for k in sorted(groups):
        runs = sorted({r["seed"]: r for r in groups[k]}.values(), key=lambda r: r["seed"])
        t = agg(runs)
        rows.append(list(k) + [",".join(str(r["seed"]) for r in runs), fa((t, False)),
                               ",".join(str(r.get("best_epoch", "")) for r in runs),
                               ",".join(sorted({r["reg_device"] for r in runs}))])
    return (["## S7: all NAVIS runs", ""]
            + table(["dataset", "mode", "global state", "cutoff", "label delay (P)", "edge weight", "seeds",
                     "test (mean ± std)", "best epoch", "regularizer on"], rows))


# ---- main ---------------------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results")
    ap.add_argument("--probe", default=None, help="default: <results>/probe")
    ap.add_argument("--loop", default=None, help="default: <results>/loop/loop_diagnostics.json")
    ap.add_argument("--heuristics", default=None, help="default: <results>/heuristics")
    ap.add_argument("--navis", default=None, help="default: <results>/navis/navis_audit_results.jsonl")
    ap.add_argument("--paper", default=os.path.join(ROOT, "reference_results", "navis", "navis_paper_reported.json"))
    ap.add_argument("--out", default=None, help="markdown file (default: <results>/tables.md)")
    ap.add_argument("--latex", default=None, help="also write LaTeX rows of Tables 4-5 and S1, S3 to this file")
    a = ap.parse_args(argv)
    probe_dir = a.probe or os.path.join(a.results, "probe")
    loop_path = a.loop or os.path.join(a.results, "loop", "loop_diagnostics.json")
    hdir = a.heuristics or os.path.join(a.results, "heuristics")
    navis_path = a.navis or os.path.join(a.results, "navis", "navis_audit_results.jsonl")
    paper = json.load(open(a.paper))
    recs = load_navis(navis_path)
    loop = load_json(loop_path, f"python scripts/audit_loop.py --out {loop_path}")
    latex = [] if a.latex else None

    parts = ["# Tables", ""]
    for lines in (t3(probe_dir, loop), t_splits(loop, latex), t_reveal(hdir), t_reeval(hdir, recs, paper, latex), t_ranks(latex),
                  s_delay_sweep(hdir, latex), s_label_vs_edge(hdir, recs, paper), s_weighting(hdir, recs),
                  s_placebo(hdir), s_navis_all(recs)):
        parts += lines + [""]
    if missing:
        parts += ["## Missing inputs", ""] + [f"- {m}" for m in dict.fromkeys(missing)]
    text = "\n".join(parts)
    print(text)
    out = a.out or os.path.join(a.results, "tables.md")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    if a.latex:
        with open(a.latex, "w", encoding="utf-8") as f:
            f.write("\n".join(latex) + "\n")


if __name__ == "__main__":
    sys.exit(main())
