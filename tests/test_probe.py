"""Tests of the label-window probe on synthetic benchmarks with a known label window.

Usage:
    python tests/test_probe.py      # or: pytest tests
"""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from label_probe import from_csv, probe, summarize  # noqa: E402
from label_probe.synthetic import make_synthetic  # noqa: E402

U = 86400


def run(data, **kw):
    res = probe(data.src, data.dst, data.ts, data.label_dict, data.label_ts, data.n_classes,
                data.n_nodes, U, weights=data.weights, **kw)
    return res, summarize(res, documented=data.documented)


def test_single_day():
    res, s = run(make_synthetic(window=(5, 5)))
    assert s.peak_offset == 5 and s.window == (5, 5) and s.availability_delay_units == 6
    assert s.window_cos > 0.999 and s.window_coverage > 0.999
    assert any("only part of the documented window" in n for n in s.notes)
    print(f"[1] label = day L+5 only        -> window {s.window}, available from L+{s.availability_delay_units:g}")


def test_multi_day():
    res, s = run(make_synthetic(window=(0, 6)))
    assert s.window == (0, 6) and s.availability_delay_units == 7
    assert any("matches the documented window" in n for n in s.notes)
    print(f"[2] label = the following week  -> window {s.window}, available from L+{s.availability_delay_units:g}")


def test_weighted_labels():
    res, s = run(make_synthetic(window=(5, 5), weighted_labels=True))
    peak = [r for r in res.rows if r["offset"] == 5][0]
    assert s.by == "abs" and peak["cos_abs"] > 0.999 and peak["cos_count"] < peak["cos_abs"]
    print(f"[3] weighted labels             -> weighting '{s.by}' (cos {peak['cos_abs']:.3f} vs "
          f"count {peak['cos_count']:.3f})")


def test_unsorted_edges():
    d = make_synthetic(window=(5, 5))
    a = run(d)[0].rows
    rng = np.random.default_rng(1)
    o = np.arange(d.ts.size)
    blocks = np.array_split(o, 50)  # shuffle within small blocks: locally unsorted edge list
    o = np.concatenate([rng.permutation(b) for b in blocks])
    d.src, d.dst, d.ts = d.src[o], d.dst[o], d.ts[o]
    d.weights = {k: v[o] for k, v in d.weights.items()}
    b = run(d)[0].rows
    assert all(abs(x["cos_count"] - y["cos_count"]) < 1e-12 for x, y in zip(a, b))
    print("[4] unsorted edge list          -> identical result (sorted copy)")


def test_csv_roundtrip():
    import pandas as pd
    d = make_synthetic(window=(5, 5))
    tmp = tempfile.mkdtemp()
    name = lambda v: f"user_{v}"  # string ids
    cls = lambda c: f"class_{c}"
    pd.DataFrame({"ts": d.ts, "src": [name(v) for v in d.src], "dst": [cls(c) for c in d.dst],
                  "w": d.weights["abs"]}).to_csv(os.path.join(tmp, "e.csv"), index=False)
    rows = [(L, name(v), cls(c), x) for L, dct in d.label_dict.items() for v, vec in dct.items()
            for c, x in enumerate(vec) if x > 0]
    pd.DataFrame(rows, columns=["ts", "node", "class", "value"]).to_csv(os.path.join(tmp, "l.csv"), index=False)
    c = from_csv(os.path.join(tmp, "e.csv"), os.path.join(tmp, "l.csv"), weight_col="w", documented=(0, 6))
    res, s = run(c)
    assert s.window == (5, 5) and s.window_cos > 0.999
    print(f"[5] CSV input with string ids   -> window {s.window}")


def test_straddled_window():
    # label day shifted by 2 hours against the unit grid (as with a time-zone offset) and sparse
    # activity: day 5 holds most of the label (highest cosine) but misses the nodes that are
    # active only in the first 2 hours of day 6
    res, s = run(make_synthetic(window=(5, 5), shift=U // 12, rate=1.0))
    rows = {r["offset"]: r for r in res.rows}
    assert s.peak_offset == 5 and rows[5]["row_coverage"] < 0.99
    assert s.window == (5, 6) and s.availability_delay_units == 7 and s.window_coverage >= 0.99
    print(f"[6] label day shifted by 2 h     -> single-offset peak 5 (coverage {rows[5]['row_coverage']:.3f}), "
          f"window {s.window}, available from L+{s.availability_delay_units:g}")


if __name__ == "__main__":
    test_single_day()
    test_multi_day()
    test_weighted_labels()
    test_unsorted_edges()
    test_csv_roundtrip()
    test_straddled_window()
    print("all tests passed")
