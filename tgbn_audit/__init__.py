"""Point-in-time audit of TGB node affinity prediction (tgbn-*), in pure NumPy.

This is the partial release that accompanies the preprint; the re-evaluation pipeline
(heuristics, blends, statistics) will be released on publication.

Main entry points:
    NodeAffinityStream: the TGB evaluation loop with the two switches of the leakage-free
        protocol (strict edge cutoff, delayed label release); Algorithm 1 of the paper.
    TGBLabels, load_tgb: loading the TGB datasets through py-tgb.
    AVAILABILITY_DELAY: availability delay of each dataset, recovered by the label-window probe.
"""
from .data import AVAILABILITY_DELAY, TGB_MA_WINDOW, TGBLabels, load_tgb, median_label_interval
from .protocol import NodeAffinityStream

__version__ = "0.9.0"
__all__ = ["AVAILABILITY_DELAY", "TGB_MA_WINDOW", "TGBLabels", "load_tgb", "median_label_interval",
           "NodeAffinityStream"]
