"""Command line of the label-window probe.

Usage:
    python -m label_probe demo                           # synthetic data, no download
    python -m label_probe tgb --name tgbn-reddit         # one TGB dataset
    python -m label_probe tgb --all                      # tgbn-trade, -genre, -reddit, -token
    python -m label_probe csv --edges e.csv --labels l.csv --unit 86400 [--weight_col w]

Writes <out_dir>/<name>.json and <name>.md, and overview.md for several datasets.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

from .core import probe, summarize
from .loaders import TGB_DOCUMENTED, auto_unit, from_csv, from_tgb, median_label_interval
from .report import overview_markdown, summary_text, table_markdown, write_outputs


def _offsets(s: str):
    """Parse "a:b" (inclusive range) or a comma list of offsets."""
    if ":" in s:
        a, b = s.split(":")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in s.split(",")]


def run_one(data, args):
    """Probe one dataset, print the result and write its outputs."""
    unit = auto_unit(data.ts, data.label_ts) if args.unit == "auto" else float(args.unit)
    n_back = int(np.sum(np.diff(data.ts) < 0))
    by = args.by if args.by in data.weights else None
    res = probe(data.src, data.dst, data.ts, data.label_dict, data.label_ts, data.n_classes,
                data.n_nodes, unit, _offsets(args.offsets), data.weights, args.n_label_times,
                args.n_rows, args.seed, by=by, max_width=args.max_width)
    s = summarize(res, documented=data.documented, coverage_threshold=args.coverage)
    print(table_markdown(res.rows, s.peak_offset))
    print()
    print(summary_text(data.name, unit, s, data.unit_label))
    if n_back:
        print(f"  (edge list had {n_back} backward time steps; a sorted copy was used)")
    print()
    meta = {"median_label_interval": median_label_interval(data.label_ts), "n_backward_steps": n_back,
            "settings": {k: v for k, v in vars(args).items() if k not in ("func",)}}
    return write_outputs(args.out_dir, data.name, unit, meta, res, s, data.unit_label)


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m label_probe", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--offsets", default="-8:8", help="'a:b' range or comma list (default -8:8)")
    common.add_argument("--unit", default="auto",
                        help="time unit u in timestamp units; 'auto' = one day for daily labels on Unix "
                             "time, else the median label interval")
    common.add_argument("--n_label_times", type=int, default=20)
    common.add_argument("--n_rows", type=int, default=300, help="labeled nodes sampled per label time")
    common.add_argument("--seed", type=int, default=0)
    common.add_argument("--coverage", type=float, default=0.99,
                        help="coverage that the label window must reach")
    common.add_argument("--by", default=None,
                        help="weighting that locates the peak (count, abs, weight); default: the one "
                             "with the highest peak cosine")
    common.add_argument("--max_width", type=int, default=7,
                        help="longest contiguous window searched around the peak, in units (0 = off)")
    common.add_argument("--out_dir", default=os.path.join("results", "probe"))
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("tgb", parents=[common], help="TGB node affinity datasets (needs py-tgb)")
    g = t.add_mutually_exclusive_group(required=True)
    g.add_argument("--name", choices=sorted(TGB_DOCUMENTED))
    g.add_argument("--all", action="store_true")
    t.add_argument("--root", default="datasets")

    c = sub.add_parser("csv", parents=[common], help="any benchmark given as edge and label CSV files")
    c.add_argument("--edges", required=True)
    c.add_argument("--labels", required=True)
    c.add_argument("--edge_cols", default="ts,src,dst", help="time, source node, destination class")
    c.add_argument("--weight_col", default=None, help="optional edge weight column")
    c.add_argument("--label_cols", default="ts,node,class,value", help="label time, node, class, value")
    c.add_argument("--name", default="csv")
    c.add_argument("--documented", default=None, help="documented window as 'first,last' offsets, e.g. 0,6")

    sub.add_parser("demo", parents=[common], help="synthetic data with a known label window")
    args = p.parse_args(argv)

    results = []
    if args.cmd == "tgb":
        names = sorted(TGB_DOCUMENTED) if args.all else [args.name]
        for n in names:
            results.append(run_one(from_tgb(n, args.root), args))
    elif args.cmd == "csv":
        doc = tuple(int(x) for x in args.documented.split(",")) if args.documented else None
        data = from_csv(args.edges, args.labels, tuple(args.edge_cols.split(",")), args.weight_col,
                        tuple(args.label_cols.split(",")), args.name, doc)
        results.append(run_one(data, args))
    else:
        from .synthetic import make_synthetic
        print("synthetic benchmark: labels are built from day L+5 only; documentation claims the next week\n")
        results.append(run_one(make_synthetic(), args))
    if len(results) > 1:
        ov = overview_markdown(results)
        print(ov)
        with open(os.path.join(args.out_dir, "overview.md"), "w", encoding="utf-8") as f:
            f.write("# Label-window probe: overview\n\n" + ov + "\n")
    return results


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
