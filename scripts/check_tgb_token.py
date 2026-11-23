"""Locate the log transform of the tgbn-token edge feature in py-tgb (Section 4.2, footnote).

The script
1. locates the installed py-tgb package (also if `tgb` is a namespace package),
2. lists every line of its Python code that takes a logarithm, and
3. lists the files of the tgbn-token dataset folder (size, modification time) with the first
   lines of each CSV, which shows whether the raw file holds amounts or their logarithms.

Usage (in the environment of the probe):
    python scripts/check_tgb_token.py
"""
import importlib.metadata as md
import importlib.util
import os
import re
import time

print("py-tgb", md.version("py-tgb"))
spec = importlib.util.find_spec("tgb.nodeproppred.dataset")
root = os.path.dirname(os.path.dirname(os.path.abspath(spec.origin)))
print("tgb package:", root)

print("\n== lines that take a logarithm ==")
pat = re.compile(r"np\.log|math\.log|log1p|log10|log2")
for d, _, fs in os.walk(root):
    if "datasets" in os.path.relpath(d, root).split(os.sep):
        continue
    for f in fs:
        if f.endswith(".py"):
            path = os.path.join(d, f)
            with open(path, encoding="utf-8", errors="ignore") as fh:
                for i, line in enumerate(fh, 1):
                    if pat.search(line):
                        print(f"{os.path.relpath(path, root)}:{i}: {line.strip()}")

print("\n== tgbn-token dataset files ==")
found = False
for d, _, fs in os.walk(root):
    if "token" not in os.path.basename(d).lower():
        continue
    found = True
    print(d)
    for f in sorted(fs):
        p = os.path.join(d, f)
        print(f"  {f}  {os.path.getsize(p):,} bytes  {time.ctime(os.path.getmtime(p))}")
        if f.endswith(".csv"):
            with open(p, encoding="utf-8", errors="ignore") as fh:
                for _ in range(3):
                    print("    |", fh.readline().rstrip())
if not found:
    print("no folder with 'token' in its name under", root)
