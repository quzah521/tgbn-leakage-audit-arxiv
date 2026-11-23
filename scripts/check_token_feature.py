"""Which transform of the tgbn-token edge feature do the labels aggregate? (Section 4.2)

py-tgb 2.3.0 returns the natural logarithm of the transferred amount when it reads its cached
files, but the raw amount on the first load after a download (scripts/check_tgb_token.py shows
where); both loaders of this repository return the logarithm in either case. At the label window
of tgbn-token (offset 5 days), this script computes the cosine between the labels and the edges
weighted by the count, by |feature| = |log amount| (the weight of the heuristics and NAVIS), and
by exp(|feature|), which equals the amount wherever the amount is at least 1. With py-tgb 2.3.0
the three cosines are 0.949, 0.965 and 1.000: the labels sum the amounts.

Usage:
    python scripts/check_token_feature.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from label_probe import from_tgb, probe  # noqa: E402


def main():
    d = from_tgb("tgbn-token")
    absfeat = np.asarray(d.weights["abs"])  # |feature| = |log amount|
    weights = {"count": d.weights["count"], "abs_feature": absfeat, "exp_abs_feature": np.exp(absfeat)}
    res = probe(d.src, d.dst, d.ts, d.label_dict, d.label_ts, d.n_classes, d.n_nodes, 86400,
                offsets=[5], weights=weights, max_width=0)
    for k, v in res.rows[0].items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")


if __name__ == "__main__":
    main()
