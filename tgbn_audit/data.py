"""Loading of the TGB node affinity datasets, and label availability per dataset.

A label stamped L (its query time) summarizes the edges of a later window. The label-window
probe (`label_probe`, Section 4.2 of the paper) recovers these windows:

    tgbn-genre    one day that straddles days L+5 and L+6   -> usable from L + 7 days
    tgbn-reddit   the single day [L+5d, L+6d)               -> usable from L + 6 days
    tgbn-token    the single day [L+5d, L+6d)               -> usable from L + 6 days
    tgbn-trade    the edges of year L                       -> usable from L + 1 year

On tgbn-genre the probe cannot tell which of the two days ends the window, so we use the later
one. On tgbn-trade, the TGB loop uses label(Y) only from the next label time Y + 1 on, so no
extra delay is needed once the strict edge cutoff is applied.
"""
from __future__ import annotations

import numpy as np

# Availability delay t_a - t_q, in units of the median label interval P
# (P = one day on genre, reddit and token, one year on trade).
AVAILABILITY_DELAY = {
    "tgbn-genre": 7.0,
    "tgbn-reddit": 6.0,
    "tgbn-token": 6.0,
    "tgbn-trade": 0.0,
}

# Window k of the MovingAverage heuristic of TGB (modules/heuristics.py), as set in
# examples/nodeproppred/<name>/moving_average.py. Despite its name, MovingAverage is a recursive
# average initialized with the node's first label, h <- ((k - 1) * h + y) / k, i.e. an exponential
# moving average with weight 1/k on the newest label. Only tgbn-genre departs from the class
# default k = 7; with k = 6 we reproduce its leaderboard values (val 0.499, test 0.509).
TGB_MA_WINDOW = {
    "tgbn-genre": 6,
    "tgbn-trade": 7,
    "tgbn-reddit": 7,
    "tgbn-token": 7,
}


class TGBLabels:
    """NumPy adapter over `tgb.nodeproppred.dataset.NodePropPredDataset`.

    It keeps the label pointer of TGB (find_next_labels_batch, return_label_ts,
    reset_label_time), so that the label times and nodes evaluated are exactly those of TGB.
    """

    def __init__(self, ds):
        self.ds = ds
        self.label_ts = np.asarray(ds.label_ts)
        self.label_dict = ds.label_dict

    def get_label_time(self):
        """Return the pending label time."""
        return self.ds.return_label_ts()

    def get_node_label(self, cur_t):
        """Return (label times, nodes, labels) of the pending label time if cur_t has passed it."""
        out = self.ds.find_next_labels_batch(cur_t)
        if out is None:
            return None
        lts, nodes, Y = out
        # float32, as in the PyG wrapper of TGB
        return (np.asarray(lts), np.asarray(nodes, dtype=np.int64), np.asarray(Y, dtype=np.float32))

    def reset(self):
        """Rewind the label pointer to the first label time."""
        self.ds.reset_label_time()


def load_tgb(name: str, root: str, weight: str):
    """Load a TGB node affinity dataset as NumPy arrays (py-tgb downloads it on first use).

    Args:
        name: one of tgbn-trade, tgbn-genre, tgbn-reddit, tgbn-token.
        root: dataset folder passed to py-tgb (py-tgb resolves it inside its package directory).
        weight: "abs" weights each edge by |edge_feat[:, 0]|, which is what NAVIS accumulates;
            "one" counts edges.

    Returns:
        A tuple (src, dst, ts, w, feat, (train, val, test) masks, TGBLabels, number of classes).
    """
    from tgb.nodeproppred.dataset import NodePropPredDataset

    ds = NodePropPredDataset(name=name, root=root, preprocess=True)
    fd = ds.full_data
    src = np.asarray(fd["sources"]).astype(np.int64)
    dst = np.asarray(fd["destinations"]).astype(np.int64)
    ts = np.asarray(fd["timestamps"]).astype(np.int64)  # as t.long() in the PyG wrapper
    feat = np.asarray(fd["edge_feat"], dtype=np.float64)
    if feat.ndim == 1:
        feat = feat[:, None]
    if name == "tgbn-token" and np.nanmax(np.abs(feat[:, 0])) > 1e3:
        # py-tgb 2.3.0 takes the log of the transferred amount only when it reads its cached
        # files, not on the first load after a download (raw amounts reach ~1e77 and overflow
        # float32). Apply the log here too, so that results do not depend on the cache.
        feat[:, 0] = np.log(feat[:, 0])
    feat = feat.astype(np.float32)
    w = np.abs(feat[:, 0]).astype(np.float64) if weight == "abs" else np.ones(src.size)
    masks = (np.asarray(ds.train_mask), np.asarray(ds.val_mask), np.asarray(ds.test_mask))
    labels = TGBLabels(ds)
    n_classes = int(ds.num_classes)
    return src, dst, ts, w, feat, masks, labels, n_classes


def median_label_interval(label_ts) -> float:
    """Median gap between consecutive distinct label times (P in the paper)."""
    lt = np.unique(np.asarray(label_ts))
    return float(np.median(np.diff(lt))) if lt.size > 1 else 1.0
