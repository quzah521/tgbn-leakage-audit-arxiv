"""Synthetic benchmark with a known label window, for the demo and the tests."""
from __future__ import annotations

import numpy as np

from .loaders import ProbeData


def make_synthetic(window=(5, 5), unit=86400, n_periods=60, n_users=150, n_classes=30,
                   rate=12.0, weighted_labels=False, seed=0, shift=0) -> ProbeData:
    """Generate users whose preferred class changes every few days.

    The label of a user at day L is the (weighted) distribution of the user's edges in
    [L + window[0]*u, L + (window[1]+1)*u), whereas the "documentation" gives the following week
    (offsets 0 to 6).

    Args:
        window: (first, last) offset of the true label window, in units.
        unit: probe unit u, in timestamp units.
        n_periods: number of units in the stream.
        n_users: number of users (labeled nodes).
        n_classes: number of classes.
        rate: mean number of edges per user and unit.
        weighted_labels: aggregate edge weights instead of counting edges.
        seed: random seed.
        shift: offset of the label window against the unit grid, in timestamp units (as when
            labels and edges use different time zones).

    Returns:
        ProbeData with the weightings "count" and "abs".
    """
    rng = np.random.default_rng(seed)
    T = n_periods * unit
    src, dst, ts, w = [], [], [], []
    for v in range(n_users):
        n_ev = rng.poisson(rate * n_periods)
        t = np.sort(rng.integers(0, T, n_ev))
        switch = np.sort(rng.uniform(0, T, rng.poisson(n_periods / 4)))
        favs = rng.integers(0, n_classes, switch.size + 1)
        f = favs[np.searchsorted(switch, t, side="right")]
        cls = np.where(rng.random(n_ev) < 0.6, f, rng.integers(0, n_classes, n_ev))
        src.append(np.full(n_ev, v)); dst.append(cls); ts.append(t)
        w.append(rng.lognormal(0.0, 1.0, n_ev))
    src, dst, ts, w = map(np.concatenate, (src, dst, ts, w))
    o = np.argsort(ts, kind="stable")
    src, dst, ts, w = src[o].astype(np.int64), dst[o].astype(np.int64), ts[o].astype(np.int64), w[o]
    label_dict = {}
    for j in range(1, n_periods - window[1] - 1):
        L = j * unit
        i0 = np.searchsorted(ts, L + window[0] * unit + shift)
        i1 = np.searchsorted(ts, L + (window[1] + 1) * unit + shift)
        dct = {}
        for v in np.unique(src[i0:i1]):
            m = src[i0:i1] == v
            vec = np.bincount(dst[i0:i1][m], weights=w[i0:i1][m] if weighted_labels else None,
                              minlength=n_classes).astype(float)
            dct[int(v)] = vec / vec.sum()
        label_dict[L] = dct
    return ProbeData("synthetic", src, dst, ts, {"count": np.ones(src.size), "abs": w}, label_dict,
                     np.array(sorted(label_dict)), n_classes, n_users, documented=(0, 6))
