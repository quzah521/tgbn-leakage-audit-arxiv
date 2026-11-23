"""Text, Markdown and JSON output of the probe."""
from __future__ import annotations

import json
import os
from dataclasses import asdict
from typing import List

from .core import ProbeResult, ProbeSummary


def unit_name(unit: float, label=None) -> str:
    """Readable name of the probe unit."""
    if label:
        return label
    return "day" if unit == 86400 else f"{unit:g} time units"


def table_markdown(rows: List[dict], peak_offset: int) -> str:
    """Markdown table of the single offsets (step 1)."""
    cols = [k for k in rows[0] if k.startswith("cos_")]
    out = ["| offset a | row coverage | " + " | ".join(cols) + " |",
           "|---|---|" + "---|" * len(cols)]
    for r in rows:
        mark = " (peak)" if r["offset"] == peak_offset else ""
        out.append(f"| {r['offset']}{mark} | {r['row_coverage']:.3f} | "
                   + " | ".join(f"{r[c]:.3f}" for c in cols) + " |")
    return "\n".join(out)


def windows_markdown(windows: List[dict], by: str, top: int = 5) -> str:
    """Markdown table of the `top` windows with the highest cosine (step 2)."""
    col = f"cos_{by}"
    best = sorted(windows, key=lambda w: -w[col])[:top]
    out = [f"| window [L+a0·u, L+(a1+1)·u) | row coverage | {col} |", "|---|---|---|"]
    out += [f"| [L+{w['first']}, L+{w['last'] + 1}) | {w['row_coverage']:.3f} | {w[col]:.3f} |" for w in best]
    return "\n".join(out)


def summary_text(name: str, unit: float, s: ProbeSummary, unit_label=None) -> str:
    """Plain-text summary of one dataset."""
    lines = [f"{name}: unit u = {unit_name(unit, unit_label)}, weighting = {s.by}",
             f"  single-offset cosine peaks at a = {s.peak_offset} (cos {s.peak_cos:.3f})",
             f"  label window ({s.method}): [L + {s.window[0]}u, L + {s.window[1] + 1}u)"
             f"  cos {s.window_cos:.3f}, row coverage {s.window_coverage:.3f}",
             f"  -> label(L) is available from L + {s.availability_delay_units:g}u"]
    if s.documented is not None:
        lines.append(f"  documented window: [L + {s.documented[0]}u, L + {s.documented[1] + 1}u)")
    lines += [f"  - {n}" for n in s.notes]
    return "\n".join(lines)


def write_outputs(out_dir: str, name: str, unit: float, meta: dict, res: ProbeResult, s: ProbeSummary,
                  unit_label=None):
    """Write <out_dir>/<name>.json and <name>.md; return the JSON content."""
    os.makedirs(out_dir, exist_ok=True)
    js = {"name": name, "unit": unit, "unit_label": unit_name(unit, unit_label), **meta, "by": res.by,
          "rows": res.rows, "windows": res.windows, "summary": asdict(s)}
    with open(os.path.join(out_dir, f"{name}.json"), "w") as f:
        json.dump(js, f, indent=2)
    md = [f"# Label-window probe: {name}", "", "```", summary_text(name, unit, s, unit_label), "```", "",
          "## Single offsets", "",
          "Cosine between the label at L and the node's edge vector in [L + a·u, L + (a+1)·u); "
          "row coverage = share of labelled nodes with at least one edge there.", "",
          table_markdown(res.rows, s.peak_offset), ""]
    if res.windows:
        md += ["## Best contiguous windows around the peak", "", windows_markdown(res.windows, s.by), ""]
    with open(os.path.join(out_dir, f"{name}.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    return js


def overview_markdown(results: List[dict]) -> str:
    """Markdown table with one row per dataset."""
    out = ["| dataset | unit | weighting | single-offset peak | label window | row coverage | available from | documented |",
           "|---|---|---|---|---|---|---|---|"]
    for js in results:
        s = js["summary"]
        doc = "—" if s["documented"] is None else f"[L+{s['documented'][0]}, L+{s['documented'][1] + 1})"
        out.append(f"| {js['name']} | {js.get('unit_label') or unit_name(js['unit'])} | {s['by']} | {s['peak_offset']} ({s['peak_cos']:.3f}) | "
                   f"[L+{s['window'][0]}, L+{s['window'][1] + 1}) | {s['window_coverage']:.3f} | "
                   f"L+{s['availability_delay_units']:g} | {doc} |")
    return "\n".join(out)
