# Temporal Leakage Distorts the TGB Node Affinity Leaderboard: A Point-in-Time Audit

This repository accompanies the preprint *Temporal Leakage Distorts the TGB Node Affinity
Leaderboard: A Point-in-Time Audit* [Experiment, Analysis & Benchmark].

## Overview

Every label of a node affinity benchmark has a **query time** t_q, at which a model must predict
it, and an **availability time** t_a, by which all interactions it summarizes have occurred. An
evaluation is point-in-time correct if, when predicting at t_q, a model has processed only edges
stamped before t_q and has received only labels with t_a ≤ t_q. The reference evaluation loop of
the [Temporal Graph Benchmark (TGB)](https://github.com/shenyangHuang/TGB) violates this on all
four node affinity datasets:

* **Early label release** (tgbn-genre, tgbn-reddit, tgbn-token). Each label summarizes a single
  day about five days after its timestamp, not the documented following week, yet the loop feeds
  it back right after scoring it, so label-based methods see five to six days ahead.
* **Same-timestamp edges** (tgbn-trade). The label of year Y is computed from the edges of year Y,
  all of which the loop processes before predicting Y; a parameter-free edge counter scores
  NDCG@10 = 1.000.
* **Batch-boundary edges** (all datasets; known as TGB issue #53). Up to two batches of later
  edges are processed before a prediction; on these datasets they carry almost no information.



## Requirements

* Python ≥ 3.9.
* NumPy ≥ 1.24, SciPy ≥ 1.10, and [py-tgb](https://pypi.org/project/py-tgb/) 2.3.0 for the
  datasets (CPU only).
* Optional: pandas ≥ 1.5 for CSV input to the probe, matplotlib ≥ 3.6 for the figures,
  pytest for the tests.

```bash
pip install -r requirements.txt
python reproduce.py test            # synthetic tests, under a minute, no data needed
```

## Datasets

py-tgb downloads each dataset on first use into its own package directory
(`<site-packages>/tgb/datasets/`), whatever the working directory. tgbn-token takes well over
10 GB and a lot of memory; on a small disk, move its folder and leave a symbolic link in its place.

py-tgb 2.3.0 returns the logarithm of the tgbn-token edge feature only when it reads its cached
files, and the raw amounts on the first load after a download (`scripts/check_tgb_token.py`).
Our loaders apply the logarithm in both cases (Section 4.2 of the paper).

## Reproducing the audit

```bash
python reproduce.py probe          # label windows      -> results/probe/
python reproduce.py loop           # loop diagnostics   -> results/loop/
python reproduce.py tables         # results/tables.md, results/tables.tex
python reproduce.py figure         # results/figures/probe.pdf, inputs.pdf, memory.pdf
```

`python reproduce.py all` runs the tests and all four steps. `--only tgbn-genre,tgbn-trade`
restricts a step to some datasets, and `--dry_run` prints the commands without running them.
The `tables` and `figure` steps use your own probe and loop outputs once they cover all four
datasets (until then, ours), and read the outputs of the re-evaluation pipeline from
`reference_results/`.
`python scripts/make_tables.py --results reference_results` rebuilds `tables.md` from our own
outputs alone.

### Running the probe

```bash
python -m label_probe demo                     # synthetic data with a known window
python -m label_probe tgb --name tgbn-reddit
python -m label_probe tgb --all --out_dir results/probe
```

For each offset a, the probe computes the cosine between the label of v at L and the edges of v
in [L + a·u, L + (a+1)·u), and the coverage, the share of labeled nodes with edges there. The
label window is the contiguous window with the highest cosine among those with coverage ≥ 0.99;
the availability time is its end.

### Notes on the setup

* **MovingAverage of TGB.** It is a recursive average initialized with the first label,
  h ← ((k−1)·h + y)/k, i.e. an exponential moving average with factor (k−1)/k, not the mean of
  the last k labels. The TGB example scripts set k = 6 on tgbn-genre and k = 7 elsewhere
  (`TGB_MA_WINDOW` in `tgbn_audit/data.py`).
* **Stream.** With `cutoff="protocol", label_delay=0`, `NodeAffinityStream` replays the TGB
  reference loop (the same batches, label times and nodes as the TGB examples and NAVIS); with
  `cutoff="strict"` and `label_delay` set to the availability delay, it is the leakage-free
  protocol.
* **Reference outputs of the pipeline** keep four decimals (`tables.md` rounds half up). NAVIS
  results cover seeds 1, 2 and 3 for every configuration in Table 4 (standard deviation at most
  0.0004). See `reference_results/README.md`.


## License

The code in this repository (`label_probe/`, `tgbn_audit/`, `scripts/`, `tests/` and
`reproduce.py`) is released under the MIT License (`LICENSE`).

