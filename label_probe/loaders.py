"""Loaders that turn a benchmark into the arrays the probe needs.

    src, dst, ts    edges (the destination is the class a label is distributed over)
    weights         {name: per-edge weight}
    label_dict      {L: {node: label vector}}, with label_ts, n_classes and n_nodes

from_tgb: the TGB node property prediction datasets (tgbn-*), through py-tgb.
from_csv: any benchmark given as two CSV files, one of edges and one of labels.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

# Window that the TGB documentation gives for each label, as (first, last) offsets in probe
# units: "the following week" on the daily datasets, "the following year" on tgbn-trade.
TGB_DOCUMENTED = {
    "tgbn-genre": (0, 6),
    "tgbn-reddit": (0, 6),
    "tgbn-token": (0, 6),
    "tgbn-trade": (0, 0),
}


@dataclass
class ProbeData:
    """Edges and labels of one benchmark, in the form the probe reads."""

    name: str
    src: np.ndarray
    dst: np.ndarray
    ts: np.ndarray
    weights: Dict[str, np.ndarray]
    label_dict: dict
    label_ts: np.ndarray
    n_classes: int
    n_nodes: int
    documented: Optional[tuple] = None
    unit_label: Optional[str] = None  # name of the probe unit if not a day, e.g. "year"


def median_label_interval(label_ts) -> float:
    """Median gap between consecutive distinct label times."""
    lt = np.unique(np.asarray(label_ts))
    return float(np.median(np.diff(lt))) if lt.size > 1 else 1.0


def auto_unit(ts, label_ts) -> float:
    """Probe unit: one day if labels are at least a day apart and timestamps are Unix seconds,
    otherwise the median label interval."""
    period = median_label_interval(label_ts)
    return 86400.0 if (period >= 86400 and np.max(ts) > 1e8) else period


def _n_nodes(src, dst, label_dict) -> int:
    """Number of node ids, counting labeled nodes that have no edges."""
    n = int(max(src.max(), dst.max())) + 1
    try:
        n = max(n, 1 + max(max(d.keys()) for d in label_dict.values() if len(d)))
    except ValueError:
        pass
    return n


def from_tgb(name: str, root: str = "datasets") -> ProbeData:
    """Load a TGB node affinity dataset; py-tgb downloads it on first use.

    py-tgb stores the datasets inside its own package directory (<site-packages>/tgb/<root>/),
    whatever the current directory. tgbn-token takes well over 10 GB there; on a small disk,
    move the dataset folder elsewhere and leave a symbolic link in its place.

    Args:
        name: one of tgbn-trade, tgbn-genre, tgbn-reddit, tgbn-token.
        root: dataset folder passed to py-tgb.

    Returns:
        ProbeData with the weightings "count" (unit weights) and "abs" (|edge feature|).
    """
    from tgb.nodeproppred.dataset import NodePropPredDataset

    ds = NodePropPredDataset(name=name, root=root, preprocess=True)
    fd = ds.full_data
    src = np.asarray(fd["sources"]).astype(np.int64)
    dst = np.asarray(fd["destinations"]).astype(np.int64)
    ts = np.asarray(fd["timestamps"]).astype(np.int64)
    feat = np.asarray(fd["edge_feat"], dtype=np.float64)
    if feat.ndim == 1:
        feat = feat[:, None]
    if name == "tgbn-token" and np.nanmax(np.abs(feat[:, 0])) > 1e3:
        # py-tgb 2.3.0 takes the log of the transferred amount only when it reads its cached
        # files, not on the first load after a download; apply it here so that results do not
        # depend on the cache
        feat[:, 0] = np.log(feat[:, 0])
    weights = {"count": np.ones(src.size), "abs": np.abs(feat[:, 0])}
    label_dict = ds.label_dict
    return ProbeData(name, src, dst, ts, weights, label_dict, np.asarray(ds.label_ts),
                     int(ds.num_classes), _n_nodes(src, dst, label_dict), TGB_DOCUMENTED.get(name),
                     "year" if name == "tgbn-trade" else None)


def from_csv(edges_path: str, labels_path: str, edge_cols=("ts", "src", "dst"), weight_col=None,
             label_cols=("ts", "node", "class", "value"), name: str = "csv",
             documented: Optional[tuple] = None) -> ProbeData:
    """Load edges and labels from two CSV files with a header row.

    Node and class ids may be strings; edges and labels share the same id spaces. Times must be
    numeric and in the same unit in both files.

    Args:
        edges_path: CSV with the columns edge_cols (time, source node, destination class) and
            optionally weight_col.
        labels_path: CSV with the columns label_cols (label time, node, class, value), one row
            per non-zero label entry.
        edge_cols: names of the time, source and destination columns of the edges.
        weight_col: optional edge weight column; adds the weighting "weight".
        label_cols: names of the time, node, class and value columns of the labels.
        name: name used in the outputs.
        documented: (first, last) offsets of the documented label window, if known.

    Returns:
        ProbeData.
    """
    import pandas as pd

    e = pd.read_csv(edges_path)
    lab = pd.read_csv(labels_path)
    t_c, s_c, d_c = edge_cols
    lt_c, n_c, c_c, v_c = label_cols
    node_codes, node_uni = pd.factorize(pd.concat([e[s_c], lab[n_c]], ignore_index=True))
    cls_codes, cls_uni = pd.factorize(pd.concat([e[d_c], lab[c_c]], ignore_index=True))
    E = len(e)
    src, dst = node_codes[:E].astype(np.int64), cls_codes[:E].astype(np.int64)
    lnode, lcls = node_codes[E:].astype(np.int64), cls_codes[E:].astype(np.int64)
    ts = e[t_c].to_numpy()
    C, N = len(cls_uni), len(node_uni)
    weights = {"count": np.ones(E)}
    if weight_col:
        weights["weight"] = np.abs(e[weight_col].to_numpy(dtype=np.float64))
    lt = lab[lt_c].to_numpy()
    val = lab[v_c].to_numpy(dtype=np.float64)
    label_dict: dict = {}
    order = np.lexsort((lnode, lt))
    for i in order:
        L, v = lt[i].item(), int(lnode[i])
        d = label_dict.setdefault(L, {})
        if v not in d:
            d[v] = np.zeros(C)
        d[v][lcls[i]] += val[i]
    return ProbeData(name, src, dst, ts, weights, label_dict, np.array(sorted(label_dict)), C, N, documented)
