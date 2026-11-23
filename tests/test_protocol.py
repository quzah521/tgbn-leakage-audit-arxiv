"""Tests of the evaluation stream on a synthetic stream (no TGB, no torch).

    stream invariants: the strict cutoff never passes an edge at or after t_q before the
    prediction at t_q, and a delayed label is never released before t_q + delay

The tests of the re-evaluation pipeline will be released with it on publication.

Usage:
    python tests/test_protocol.py      # or: pytest tests
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from tgbn_audit import NodeAffinityStream  # noqa: E402


class FakeLabels:
    """Label source with the pointer semantics of NodePropPredDataset.find_next_labels_batch."""

    def __init__(self, label_dict):
        self.label_dict = label_dict
        self.label_ts = np.sort(np.array(list(label_dict.keys())))
        self.idx = 0

    def get_label_time(self):
        return self.label_ts[-1] if self.idx >= self.label_ts.size else self.label_ts[self.idx]

    def get_node_label(self, cur_t):
        if self.idx >= self.label_ts.size:
            return None
        ts = self.label_ts[self.idx]
        if cur_t >= ts:
            self.idx += 1
            dct = self.label_dict[ts]
            nodes = np.array(list(dct.keys()))
            Y = np.stack([dct[k] for k in dct]).astype(np.float32)
            return np.full(nodes.size, ts, dtype=int), nodes.astype(np.int64), Y
        return None

    def reset(self):
        self.idx = 0


def synth(timing: bool, seed=0, U=200, C=40, P=1000, n_periods=80, rate=20.0, p_fav=0.6,
          label_offset=0):
    """Synthetic stream in which users switch their favorite class.

    With timing=True, users switch at random moments within periods; with timing=False, only at
    period starts. The label at L_j is the normalized count vector of the user's edges in
    [L_j + label_offset*P, L_j + (label_offset+1)*P).
    """
    rng = np.random.default_rng(seed)
    T_end = n_periods * P
    src, dst, ts = [], [], []
    for u in range(U):
        bg = rng.dirichlet(np.full(C, 0.3))
        n_sw = rng.poisson(T_end / (2 * P))
        sw = np.sort(rng.uniform(0, T_end, n_sw))
        if not timing:
            sw = np.unique(np.floor(sw / P) * P)
        favs = rng.integers(0, C, sw.size + 1)
        n_ev = rng.poisson(rate * n_periods)
        t = np.sort(rng.integers(0, T_end, n_ev))
        f = favs[np.searchsorted(sw, t, side="right")]
        cls = np.where(rng.random(n_ev) < p_fav, f, rng.choice(C, n_ev, p=bg))
        src.append(np.full(n_ev, C + u))
        dst.append(cls)
        ts.append(t)
    src, dst, ts = map(np.concatenate, (src, dst, ts))
    o = np.argsort(ts, kind="stable")
    src, dst, ts = src[o].astype(np.int64), dst[o].astype(np.int64), ts[o].astype(np.int64)
    label_dict = {}
    for j in range(1, n_periods - label_offset):
        L = j * P
        a = L + label_offset * P
        i0, i1 = np.searchsorted(ts, a), np.searchsorted(ts, a + P)
        dct = {}
        for u in np.unique(src[i0:i1]):
            m = src[i0:i1] == u
            v = np.bincount(dst[i0:i1][m], minlength=C).astype(float)
            dct[int(u)] = v / v.sum()
        label_dict[L] = dct
    q1, q2 = np.quantile(ts, [0.70, 0.85])
    masks = (ts <= q1, (ts > q1) & (ts <= q2), ts > q2)
    return src, dst, ts, np.ones(src.size), masks, FakeLabels(label_dict), C


def test_stream_invariants():
    src, dst, ts, w, masks, labels, C = synth(True)
    splits = [np.flatnonzero(m) for m in masks]
    delay = 2500
    for cutoff in ["strict", "protocol"]:
        stream = NodeAffinityStream(ts, labels, 200, cutoff, label_delay=delay)
        stream.reset()
        max_seen, n_pred, n_rel, worst_rel = -np.inf, 0, 0, np.inf
        late_edges, released_since = 0, []
        for idx, ph in zip(splits, ["train", "val", "test"]):
            for ev in stream.run_split(idx, ph):
                if ev[0] == "edges":
                    max_seen = max(max_seen, ts[ev[1]].max())
                elif ev[0] == "predict":
                    n_pred += 1
                    late_edges += int(max_seen >= ev[1])
                    # every label released before this prediction must be available by now
                    for t_lab in released_since:
                        worst_rel = min(worst_rel, ev[1] - (t_lab + delay))
                    released_since = []
                else:
                    n_rel += 1
                    released_since.append(ev[1])
        print(f"stream invariants, {cutoff:8s}: {n_pred} predictions, {late_edges} with an edge at/after t_q "
              f"already ingested; labels released {n_rel}, min(t_release - (t_q + delay)) = {worst_rel}")
        if cutoff == "strict":
            assert late_edges == 0
        assert worst_rel >= 0


if __name__ == "__main__":
    test_stream_invariants()
    print("all tests passed")
