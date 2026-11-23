"""Model-agnostic evaluation stream for TGB node affinity prediction (Section 3.2-3.3).

Every label has a query time t_q (its timestamp, at which it is predicted) and an availability
time t_a (the end of the window it summarizes). An evaluation is point-in-time correct if,
whenever a model predicts at t_q,

    (i)  it has processed only edges with t < t_q, and
    (ii) it has received as inputs only labels with t_a <= t_q.

`NodeAffinityStream` replays the reference loop of TGB (the TGB examples and NAVIS use the same
loop): edges arrive in batches of `batch_size`, and the first batch whose last edge is later than
the pending label time L_i triggers the prediction of the labels stamped at L_i. The two parts of
the leakage-free protocol are switches:

    cutoff="protocol"  TGB reference loop. Inside the triggering batch, every edge with
                       t < L_{i+1} is processed before L_i is predicted, which includes edges at
                       or after L_i (on tgbn-trade, every edge of the target year).
    cutoff="strict"    only edges with t < L_i are processed before L_i is predicted; the other
                       edges are held back and processed after the prediction.
    label_delay=d      a label scored at L is released as a model input only before the first
                       prediction at a time >= L + d. TGB: d = 0 (released right after scoring).
                       Leakage-free: d = t_a - t_q (see data.AVAILABILITY_DELAY).

The stream yields events and leaves it to the caller what to do with them:

    ("edges", idx)               process the edges idx (indices into the full edge arrays)
    ("predict", t_q, nodes, Y)   predict `nodes` at t_q; Y is the target, to be used only for
                                 the metric or the loss, never as an input before scoring
    ("release", t_q, nodes, Y)   the labels scored at t_q may now be used as inputs

Example:

    stream = NodeAffinityStream(ts, TGBLabels(dataset), cutoff="strict",
                                label_delay=AVAILABILITY_DELAY[name] * P)
    for split, phase in [(train_idx, "train"), (val_idx, "val"), (test_idx, "test")]:
        for ev in stream.run_split(split, phase):
            if ev[0] == "edges":
                model.update(src[ev[1]], dst[ev[1]], ts[ev[1]], msg[ev[1]])
            elif ev[0] == "predict":
                _, t_q, nodes, Y = ev
                scores = model.predict(nodes, t_q)          # then NDCG or loss against Y
            else:
                _, t_q, nodes, Y = ev
                model.observe_labels(nodes, Y)              # label-based models only

As in TGB, the stream keeps its state (held-back edges, label pointer, queued labels) across
the three splits; call reset() before an independent pass.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Iterator, List, Tuple

import numpy as np


class NodeAffinityStream:
    """TGB evaluation loop with a strict edge cutoff and delayed label release as switches.

    Besides the events, the stream records edge-side diagnostics per label time:
    `leak` (number of edges with t >= L_i processed before predicting L_i) and
    `now_minus_lt` (latest processed edge time minus L_i; >= 0 means edge-side leakage).
    """

    def __init__(self, ts: np.ndarray, labels, batch_size: int = 200, cutoff: str = "strict",
                 label_delay: float = 0.0):
        """
        Args:
            ts: (n_edges,) edge timestamps in loader order (the order of TGB).
            labels: object with get_label_time(), get_node_label(t) and reset(), e.g.
                data.TGBLabels, which wraps the NodePropPredDataset of TGB.
            batch_size: edges per batch (200 in the TGB examples and in NAVIS).
            cutoff: "protocol" (TGB reference loop) or "strict".
            label_delay: release delay in the time unit of `ts` (e.g. 6 * 86400 on tgbn-reddit).
        """
        if cutoff not in ("protocol", "strict"):
            raise ValueError(f"cutoff must be 'protocol' or 'strict', got {cutoff!r}")
        self.ts = np.asarray(ts)
        self.labels = labels
        self.B = int(batch_size)
        self.strict = cutoff == "strict"
        self.label_delay = float(label_delay)
        self._reset_state()

    def _reset_state(self) -> None:
        self.buf = None                 # edge indices held back by the strict cutoff
        self.pending: List[tuple] = []  # queued labels (release_time, t_q, nodes, Y), FIFO
        self.stopped = False
        self.now = -np.inf              # latest processed edge time
        self.leak: List[int] = []
        self.now_minus_lt: List[float] = []
        self.n_label_times = defaultdict(int)
        self._ahead = 0

    def reset(self) -> None:
        """Rewind the label pointer of TGB and clear all stream state."""
        self.labels.reset()
        self._reset_state()

    def _edges(self, ind: np.ndarray, pending_lt) -> Iterator[Tuple]:
        if ind.size == 0:
            return
        t = self.ts[ind]
        self._ahead += int(np.sum(t >= pending_lt))
        self.now = max(self.now, float(t.max()))
        yield ("edges", ind)

    def _release(self, t_pred) -> Iterator[Tuple]:
        while self.pending and self.pending[0][0] <= t_pred:
            _, lt0, nodes, Y = self.pending.pop(0)
            yield ("release", lt0, nodes, Y)

    def run_split(self, idx: np.ndarray, phase: str = "") -> Iterator[Tuple]:
        """Yield the events of one split.

        Args:
            idx: indices of the edges of the split, in loader order.
            phase: name of the split ("train", "val" or "test"), used only for counting.

        Yields:
            ("edges", idx), ("predict", t_q, nodes, Y) and ("release", t_q, nodes, Y) events.
        """
        if self.stopped:
            return
        ts = self.ts
        idx = np.asarray(idx)
        label_t = self.labels.get_label_time()
        for b0 in range(0, idx.size, self.B):
            ind = idx[b0:b0 + self.B]
            query_t = ts[ind[-1]]
            if self.buf is not None:
                ind = np.concatenate([self.buf, ind])
                self.buf = None
            t = ts[ind]
            if query_t > label_t:
                lab = self.labels.get_node_label(query_t)
                if lab is None:  # no labels left: the TGB and NAVIS loops stop here
                    self.stopped = True
                    return
                lts, nodes, Y = lab
                lt = int(lts[0])
                label_t = self.labels.get_label_time()  # the pointer has moved to L_{i+1}
                m = t < (lt if self.strict else label_t)
                yield from self._edges(ind[m], lt)
                self.leak.append(self._ahead)
                self._ahead = 0
                ind, t = ind[~m], t[~m]
                self.now_minus_lt.append(float(self.now - lt))
                yield from self._release(lt)
                yield ("predict", lt, nodes, Y)
                self.n_label_times[phase] += 1
                if self.label_delay <= 0:
                    yield ("release", lt, nodes, Y)
                else:
                    self.pending.append((lt + self.label_delay, lt, nodes, Y))
            if self.strict:
                m = t < label_t
                yield from self._edges(ind[m], label_t)
                if not m.all():
                    self.buf = ind[~m]
            else:
                yield from self._edges(ind, label_t)
