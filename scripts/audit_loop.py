"""Edge-side diagnostics of the TGB reference loop (Table 3; S1 for all splits).

For each dataset, edge cutoff (protocol = TGB reference loop, strict = leakage-free) and split,
the loop is replayed with NodeAffinityStream, without a model, and per query time t_q we count

    n_eq   edges stamped exactly at t_q that were processed before predicting t_q
    n_gt   edges stamped after t_q that were processed before predicting t_q
    delta  latest processed edge time - t_q (< 0: no edge-side leakage; 0: same timestamp only)

summarized per split (mean, max), with the mean number of edges per label interval for context.
Label-side leakage needs no replay: under the TGB loop its look-ahead is a* x Delta (Section 3.2).

Usage (from the repository root). Results are merged into --out, so the datasets can run one
per process, which keeps the memory of tgbn-reddit and tgbn-token apart:
    python scripts/audit_loop.py --names tgbn-trade
    python scripts/audit_loop.py --names tgbn-genre
    python scripts/audit_loop.py --names tgbn-reddit
    python scripts/audit_loop.py --names tgbn-token
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from tgbn_audit.protocol import NodeAffinityStream  # noqa: E402

PHASES = ("train", "val", "test")


def loop_diagnostics(ts, labels, splits, period, cutoff="protocol", batch_size=200) -> dict:
    """Replay the loop over the three splits, carrying its state across them as TGB does.

    Args:
        ts: (E,) edge timestamps.
        labels: label source (data.TGBLabels).
        splits: edge indices of the train, validation and test split.
        period: median label interval P.
        cutoff: "protocol" or "strict".
        batch_size: edges per batch.

    Returns:
        {split: summary of n_eq, n_gt and delta} and a cross-check against the stream's counter.
    """
    ts = np.asarray(ts)
    stream = NodeAffinityStream(ts, labels, batch_size, cutoff, label_delay=0.0)
    stream.reset()
    since_pred = []      # edge times processed since the previous prediction
    latest = -np.inf     # latest processed edge time
    out = {}
    for idx, phase in zip(splits, PHASES):
        n_eq, n_gt, delta = [], [], []
        for ev in stream.run_split(idx, phase):
            if ev[0] == "edges":
                t = ts[ev[1]]
                since_pred.append(t)
                latest = max(latest, float(t.max()))
            elif ev[0] == "predict":
                t_q = ev[1]
                # an edge stamped at or after t_q cannot have been processed before the previous
                # prediction (the loop never passes edges at or after the next label time), so it
                # suffices to count the edges processed since then
                t = np.concatenate(since_pred) if since_pred else np.empty(0)
                n_eq.append(int(np.sum(t == t_q)))
                n_gt.append(int(np.sum(t > t_q)))
                delta.append(latest - float(t_q))
                since_pred = []
        tt = ts[idx]
        n_int = max((tt.max() - tt.min()) / period, 1.0) if tt.size else 1.0
        n_eq_a, n_gt_a, d_a = np.array(n_eq), np.array(n_gt), np.array(delta)
        out[phase] = {
            "query_times": int(n_eq_a.size),
            "mean_n_eq": float(n_eq_a.mean()) if n_eq_a.size else 0.0,
            "mean_n_gt": float(n_gt_a.mean()) if n_gt_a.size else 0.0,
            "max_n_gt": int(n_gt_a.max()) if n_gt_a.size else 0,
            "share_queries_with_leak": float(np.mean((n_eq_a + n_gt_a) > 0)) if n_eq_a.size else 0.0,
            "max_delta": float(d_a.max()) if d_a.size else float("nan"),
            "edges_per_label_interval": float(tt.size / n_int),
        }
    # cross-check against the stream's own counter (edges at/after t_q, all splits)
    leak = np.asarray(stream.leak, dtype=float)
    out["check_stream_counter_mean"] = float(leak.mean()) if leak.size else 0.0
    return out


def main(argv=None):
    from tgbn_audit.data import load_tgb, median_label_interval

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--names", default="tgbn-trade,tgbn-genre,tgbn-reddit,tgbn-token")
    ap.add_argument("--cutoffs", default="protocol,strict")
    ap.add_argument("--root", default="datasets")
    ap.add_argument("--batch_size", type=int, default=200)
    ap.add_argument("--out", default=os.path.join("results", "loop", "loop_diagnostics.json"))
    a = ap.parse_args(argv)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    res = {}
    if os.path.exists(a.out):  # merge with datasets from earlier runs
        with open(a.out) as f:
            res = json.load(f)
    for name in [n.strip() for n in a.names.split(",") if n.strip()]:
        src, dst, ts, w, feat, masks, labels, C = load_tgb(name, a.root, "one")
        splits = [np.flatnonzero(m) for m in masks]
        period = median_label_interval(labels.label_ts)
        res[name] = {"period": period}
        for cutoff in [c.strip() for c in a.cutoffs.split(",") if c.strip()]:
            r = loop_diagnostics(ts, labels, splits, period, cutoff, a.batch_size)
            res[name][cutoff] = r
            for ph in PHASES:
                x = r[ph]
                print(f"{name:12s} {cutoff:8s} {ph:5s}  queries {x['query_times']:5d}  "
                      f"n_eq {x['mean_n_eq']:10.1f}  n_gt {x['mean_n_gt']:7.1f} (max {x['max_n_gt']:3d})  "
                      f"max delta {x['max_delta']:12.6g}  edges/interval {x['edges_per_label_interval']:12.1f}",
                      flush=True)
        labels.reset()
        with open(a.out, "w") as f:
            json.dump(res, f, indent=2)
        del src, dst, ts, w, feat, masks, labels, splits
        gc.collect()
    print("written to", a.out)


if __name__ == "__main__":
    main()
