"""Label-window probe: which time window does each label actually summarize? (Section 3.1)

Many dynamic-graph benchmarks attach to a time L a label that aggregates a node's future
interactions, e.g. "the distribution of the node's edges over the next week". The probe recovers,
from the labels and the edges alone, the window that a label is computed from.

Step 1 (single offsets). For a sample of label times L and labeled nodes v, and for each
integer offset a, compute

    cos_a = cosine(label(v, L), edge vector of v in [L + a*u, L + (a+1)*u))

where u is a time unit (one day for daily labels) and the edge vector holds the (weighted)
edges of v to each class; also record the coverage, the share of labeled nodes with at least one
edge in that unit. The cosine peaks at the offsets the label comes from.

Step 2 (window search). Around the peak, compute the same quantities for every contiguous
window [L + a0*u, L + (a1+1)*u) of at most max_width units. A node has a label only if it has
edges in the label window, so the label window must reach full coverage: the probe takes the
window with the highest cosine among those whose coverage reaches the threshold.

For the evaluation this gives
    t_q = L                        query time (when the label is predicted)
    t_a = L + (a1 + 1) * u         availability time (when its content is complete)
A model that receives label(L) as an input before t_a uses information from the future.

Pure NumPy; independent of any model or learning library.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np


@dataclass
class ProbeResult:
    """Raw output of `probe`.

    Attributes:
        rows: one dict per single offset a, with offset, row_coverage and cos_<weighting>.
        windows: one dict per contiguous window around the peak (step 2), with first, last,
            row_coverage and cos_<weighting>.
        by: weighting used to locate the peak.
    """

    rows: List[dict]
    windows: List[dict] = field(default_factory=list)
    by: Optional[str] = None


def _prepare(src, dst, ts, weights):
    """Convert inputs to arrays and sort the edges by time if needed."""
    src = np.asarray(src, dtype=np.int64)
    dst = np.asarray(dst, dtype=np.int64)
    ts = np.asarray(ts)
    if weights is None:
        weights = {"count": np.ones(src.size)}
    weights = {k: np.asarray(v, dtype=np.float64) for k, v in weights.items()}
    if np.any(np.diff(ts) < 0):  # edge lists are not always sorted by time; sort a copy
        o = np.argsort(ts, kind="stable")
        src, dst, ts = src[o], dst[o], ts[o]
        weights = {k: v[o] for k, v in weights.items()}
    return src, dst, ts, weights


def _sample(label_dict, label_ts, n_label_times, n_rows, seed):
    """Sample label times from the middle 80% of all label times, and up to n_rows nodes each."""
    rng = np.random.default_rng(seed)
    lt_all = np.sort(np.asarray(label_ts))
    lo, hi = int(0.1 * lt_all.size), int(0.9 * lt_all.size)
    pool = lt_all[lo:hi] if hi - lo >= n_label_times else lt_all
    picks = rng.choice(pool, size=min(n_label_times, pool.size), replace=False)
    out = []
    for L in picks:
        dct = label_dict[L]
        keys = np.fromiter(dct.keys(), dtype=np.int64)
        sel = keys if keys.size <= n_rows else rng.choice(keys, n_rows, replace=False)
        Y = np.stack([np.asarray(dct[int(k)], dtype=np.float64) for k in sel])
        out.append((L, sel, Y, np.linalg.norm(Y, axis=1)))
    return out


def _window_scores(src, dst, ts, weights, C, n_nodes, samples, lo_off, hi_off):
    """Mean cosine (per weighting) and coverage of the windows [L + lo_off, L + hi_off)."""
    acc = {wn: [] for wn in weights}
    cov = []
    for L, sel, Y, yn in samples:
        pos = np.full(n_nodes, -1, dtype=np.int64)
        pos[sel] = np.arange(sel.size)
        i0 = np.searchsorted(ts, L + lo_off, side="left")
        i1 = np.searchsorted(ts, L + hi_off, side="left")
        p = pos[src[i0:i1]]
        k = p >= 0
        for wn, wv in weights.items():
            M = np.bincount(p[k] * C + dst[i0:i1][k], weights=wv[i0:i1][k],
                            minlength=sel.size * C).reshape(sel.size, C)
            den = np.linalg.norm(M, axis=1) * yn
            cos = np.divide((M * Y).sum(1), den, out=np.zeros(sel.size), where=den > 0)
            acc[wn].append(cos.mean())
        cov.append(np.unique(p[k]).size / sel.size)
    r = {"row_coverage": float(np.mean(cov))}
    for wn in weights:
        r[f"cos_{wn}"] = float(np.mean(acc[wn]))
    return r


def probe(src, dst, ts, label_dict: Mapping, label_ts, n_classes: int, n_nodes: int, unit: float,
          offsets: Sequence[int] = tuple(range(-8, 9)), weights: Optional[Mapping[str, np.ndarray]] = None,
          n_label_times: int = 20, n_rows: int = 300, seed: int = 0, by: Optional[str] = None,
          max_width: int = 7) -> ProbeResult:
    """Run both steps of the probe.

    Args:
        src, dst, ts: (E,) source node, destination class and timestamp of each edge.
        label_dict: {L: {node: (n_classes,) label vector}}.
        label_ts: label times (the keys of label_dict).
        n_classes: number of classes.
        n_nodes: number of node ids.
        unit: time unit u, in timestamp units.
        offsets: offsets a of step 1, in units.
        weights: {name: (E,) edge weights}, one cosine column each (default: count edges).
        n_label_times: label times sampled.
        n_rows: labeled nodes sampled per label time.
        seed: seed of the sampling.
        by: weighting that locates the peak (default: the one with the highest peak cosine).
        max_width: longest window searched in step 2, in units (0 skips step 2).

    Returns:
        ProbeResult with the single offsets and the searched windows.
    """
    src, dst, ts, weights = _prepare(src, dst, ts, weights)
    C = int(n_classes)
    samples = _sample(label_dict, label_ts, n_label_times, n_rows, seed)
    rows = []
    for a in offsets:
        r = {"offset": int(a)}
        r.update(_window_scores(src, dst, ts, weights, C, n_nodes, samples, a * unit, (a + 1) * unit))
        rows.append(r)
    if by is None:
        by = max(weights, key=lambda w: max(r[f"cos_{w}"] for r in rows))
    peak = max(rows, key=lambda r: r[f"cos_{by}"])["offset"]
    windows = []
    for a0 in range(peak - max_width + 1, peak + 1):  # empty when max_width <= 0
        for a1 in range(peak, a0 + max_width):
            w = {"first": a0, "last": a1}
            w.update(_window_scores(src, dst, ts, weights, C, n_nodes, samples, a0 * unit, (a1 + 1) * unit))
            windows.append(w)
    return ProbeResult(rows, windows, by)


def probe_label_windows(src, dst, ts, label_dict, label_ts, n_classes, n_nodes, unit,
                        offsets=tuple(range(-8, 9)), weights=None, n_label_times=20, n_rows=300, seed=0):
    """Run step 1 only; returns one dict per offset with row_coverage and cos_<weighting>."""
    return probe(src, dst, ts, label_dict, label_ts, n_classes, n_nodes, unit, offsets, weights,
                 n_label_times, n_rows, seed, max_width=0).rows


@dataclass
class ProbeSummary:
    """Label window read off a ProbeResult.

    Attributes:
        by: weighting used.
        peak_offset: single offset with the highest cosine.
        peak_cos: cosine at that offset.
        window: (first, last) offset of the label window.
        window_cos: cosine of the label window.
        window_coverage: coverage of the label window.
        availability_delay_units: t_a - t_q in units of u (= last + 1).
        method: how the window was chosen ("window search ..." or "coverage rule ...").
        documented: (first, last) offsets of the documented window, if known.
        notes: readable remarks (agreement with the documentation, look-ahead).
    """

    by: str
    peak_offset: int
    peak_cos: float
    window: tuple
    window_cos: float
    window_coverage: float
    availability_delay_units: float
    method: str
    documented: Optional[tuple] = None
    notes: List[str] = field(default_factory=list)


def summarize(result: ProbeResult, documented: Optional[tuple] = None,
              coverage_threshold: float = 0.99) -> ProbeSummary:
    """Read the label window off a ProbeResult.

    With a window search (step 2), the label window is the contiguous window with the highest
    cosine among those whose coverage reaches coverage_threshold. Without one, it is the peak
    offset extended by the adjacent offsets whose coverage reaches the threshold.

    Args:
        result: output of `probe`.
        documented: (first, last) offsets of the documented window, for the notes.
        coverage_threshold: coverage that a label window must reach.

    Returns:
        ProbeSummary.
    """
    rows = result.rows
    by = result.by or max((k[4:] for k in rows[0] if k.startswith("cos_")),
                          key=lambda w: max(r[f"cos_{w}"] for r in rows))
    col = f"cos_{by}"
    pk = max(rows, key=lambda r: r[col])
    peak = pk["offset"]
    notes = []
    if result.windows:
        # a label exists only for a node with edges in its window, so the label window must
        # cover (almost) every labeled node: take the best window among those that do
        full = [w for w in result.windows if w["row_coverage"] >= coverage_threshold]
        best = max(full or result.windows, key=lambda w: w[col])
        window, wcos, wcov = (best["first"], best["last"]), best[col], best["row_coverage"]
        method = ("window search (highest cosine among windows with full row coverage)" if full
                  else "window search (no window reaches full row coverage)")
    else:
        by_off = {r["offset"]: r for r in rows}
        a0 = a1 = peak
        while (a0 - 1) in by_off and by_off[a0 - 1]["row_coverage"] >= coverage_threshold:
            a0 -= 1
        while (a1 + 1) in by_off and by_off[a1 + 1]["row_coverage"] >= coverage_threshold:
            a1 += 1
        # single offsets only: the coverage of the union is at least the largest single coverage
        wcov = max(by_off[a]["row_coverage"] for a in range(a0, a1 + 1))
        window, wcos, method = (a0, a1), pk[col], "coverage rule (single offsets; cos = peak offset)"
    if wcov < coverage_threshold:
        notes.append(f"row coverage of the label window is {wcov:.3f} < {coverage_threshold}: "
                     f"labels may also use data outside the searched windows")
    delay = float(window[1] + 1)
    if documented is not None:
        d0, d1 = documented
        if window == (d0, d1):
            notes.append("label window matches the documented window")
        elif d0 <= window[0] and window[1] <= d1:
            notes.append(f"label window covers only part of the documented window {documented}")
        else:
            notes.append(f"label window lies outside the documented window {documented}")
    if delay > 0:
        notes.append(f"label(L) is complete only at L + {delay:g} units; using it as an input "
                     f"before then is look-ahead")
    return ProbeSummary(by, int(peak), float(pk[col]), window, float(wcos), float(wcov), delay,
                        method, documented, notes)
