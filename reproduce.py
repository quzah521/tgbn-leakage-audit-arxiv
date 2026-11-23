"""Reproduce the audit results of the paper (Linux, macOS and Windows); partial release.

This release covers the audit itself (Section 4: label windows, availability delays and loop
diagnostics) and rebuilds every table and figure from the reference outputs. The re-evaluation
pipeline of Sections 5 and 6 (heuristics, blends, NAVIS patch) will be released on publication;
until then, the tables and figures read its outputs from reference_results/.

Usage:
    python reproduce.py test                         # synthetic checks, under a minute, no data
    python reproduce.py probe                        # label windows (Table 3, Figure 2)
    python reproduce.py loop                         # loop diagnostics (Table 3, S1)
    python reproduce.py tables                       # results/tables.md, results/tables.tex
    python reproduce.py figure                       # Figures 2-4 in results/figures/
    python reproduce.py all                          # test, probe, loop, tables, figure

Options: --only tgbn-genre,tgbn-reddit restricts a step to some datasets; --dry_run prints the
commands without running them. py-tgb downloads each dataset on first use; tgbn-token needs well
over 10 GB of disk and a lot of memory. Every step writes to results/; reference_results/ holds
our own outputs.
"""
import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
DATASETS = ["tgbn-trade", "tgbn-genre", "tgbn-reddit", "tgbn-token"]
RES = os.path.join(ROOT, "results")
REF = os.path.join(ROOT, "reference_results")


def run(cmd, cwd=ROOT, dry=False):
    """Print a command and run it (unless dry); stop on the first failure."""
    print("$", " ".join(cmd), flush=True)
    if not dry:
        subprocess.run(cmd, cwd=cwd, check=True)


def wanted(name, only):
    """True if the dataset is selected by --only (all datasets by default)."""
    return only is None or name in only


def step_test(a):
    run([PY, os.path.join("tests", "test_probe.py")], dry=a.dry_run)
    run([PY, os.path.join("tests", "test_protocol.py")], dry=a.dry_run)


def step_probe(a):
    names = [n for n in DATASETS if wanted(n, a.only)]
    args = ["--all"] if len(names) == len(DATASETS) else ["--name", names[0]] if len(names) == 1 else None
    if args is None:  # several but not all: one process each
        for n in names:
            run([PY, "-m", "label_probe", "tgb", "--name", n, "--out_dir", os.path.join(RES, "probe")], dry=a.dry_run)
        return
    run([PY, "-m", "label_probe", "tgb", *args, "--out_dir", os.path.join(RES, "probe")], dry=a.dry_run)


def step_loop(a):
    for n in DATASETS:  # one process per dataset keeps the memory of the large datasets apart
        if wanted(n, a.only):
            run([PY, os.path.join("scripts", "audit_loop.py"), "--names", n,
                 "--out", os.path.join(RES, "loop", "loop_diagnostics.json")], dry=a.dry_run)


def probe_dir():
    """results/probe/ if it holds the probe of all four datasets, else reference_results/probe/."""
    p = os.path.join(RES, "probe")
    done = all(os.path.exists(os.path.join(p, f"{n}.json")) for n in DATASETS)
    return p if done else os.path.join(REF, "probe")


def loop_file():
    """results/loop/loop_diagnostics.json if it covers all four datasets, else the reference file."""
    p = os.path.join(RES, "loop", "loop_diagnostics.json")
    if os.path.exists(p):
        import json
        with open(p, encoding="utf-8") as f:
            if all(n in json.load(f) for n in DATASETS):
                return p
    return os.path.join(REF, "loop", "loop_diagnostics.json")


def step_tables(a):
    # probe and loop from your own runs once they cover all four datasets (else from
    # reference_results/); heuristics and NAVIS always from reference_results/
    run([PY, os.path.join("scripts", "make_tables.py"), "--results", RES,
         "--probe", probe_dir(), "--loop", loop_file(),
         "--heuristics", os.path.join(REF, "heuristics"),
         "--navis", os.path.join(REF, "navis", "navis_audit_results.jsonl"),
         "--latex", os.path.join(RES, "tables.tex")], dry=a.dry_run)


def step_figure(a):
    fig = os.path.join(RES, "figures")
    run([PY, os.path.join("scripts", "plot_probe.py"), "--results", probe_dir(),
         "--out", os.path.join(fig, "probe.pdf"), "--png", os.path.join(fig, "probe.png")], dry=a.dry_run)
    # Figures 3 and 4 show outputs of the re-evaluation pipeline, read from reference_results/
    run([PY, os.path.join("scripts", "plot_inputs.py"), "--results", REF,
         "--out", os.path.join(fig, "inputs.pdf"), "--png", os.path.join(fig, "inputs.png")], dry=a.dry_run)
    run([PY, os.path.join("scripts", "plot_memory.py"), "--results", REF,
         "--out", os.path.join(fig, "memory.pdf"), "--png", os.path.join(fig, "memory.png")], dry=a.dry_run)


STEPS = {"test": step_test, "probe": step_probe, "loop": step_loop, "tables": step_tables,
         "figure": step_figure}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=list(STEPS) + ["all"])
    ap.add_argument("--only", default=None, help="comma list of datasets (default: all four)")
    ap.add_argument("--dry_run", action="store_true")
    a = ap.parse_args()
    a.only = None if a.only is None else {n.strip() for n in a.only.split(",") if n.strip()}
    steps = ["test", "probe", "loop", "tables", "figure"] if a.step == "all" else [a.step]
    for s in steps:
        print(f"\n== {s} ==", flush=True)
        STEPS[s](a)


if __name__ == "__main__":
    main()
