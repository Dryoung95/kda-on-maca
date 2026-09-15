#!/usr/bin/env python3
"""Full kernel inventory of a mcTracer trace.

No upstream equivalent — ncu is replay-based and profiles one kernel at a
time, so it never needed a whole-run inventory. mcTracer records every
launch in one pass, so this view is cheap and often the first thing worth
looking at: which kernels run, how often, and how much time each takes.

Usage:
    python3 kernel_inventory.py --run-dir $PROFILE_RUN_DIR \
        --report reports/full_v1.json --tag v1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mctracer_utils import action_name, launch_geometry, load_report  # noqa: E402


def inventory(report_path):
    _, actions = load_report(report_path)
    agg = {}
    order = []
    for a in actions:
        name = action_name(a) or "?"
        if name not in agg:
            agg[name] = {"count": 0, "total_us": 0.0, "max_us": 0.0, "min_us": None,
                         "example": a}
            order.append(name)
        rec = agg[name]
        dur_us = (a.get("dur", 0) or 0) / 1000.0
        rec["count"] += 1
        rec["total_us"] += dur_us
        rec["max_us"] = max(rec["max_us"], dur_us)
        rec["min_us"] = dur_us if rec["min_us"] is None else min(rec["min_us"], dur_us)
    return [(n, agg[n]) for n in order], actions


def render(rows, actions, tag, top=25):
    total = sum(r[1]["total_us"] for r in rows) or 1.0
    lines = [f"# Kernel inventory — {tag}", "",
             f"distinct kernels: {len(rows)}   total launches: {len(actions)}   "
             f"total device time: {total:,.1f} µs", ""]
    lines.append("| Kernel | Launches | Total µs | Share % | Avg µs | Max µs | Min µs | Grid | Block | Regs |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---|---|---:|")
    ranked = sorted(rows, key=lambda r: -r[1]["total_us"])
    for name, rec in ranked[:top]:
        ex = rec["example"]
        geo = launch_geometry(ex)
        short = name if len(name) <= 52 else name[:49] + "..."
        lines.append(
            f"| {short} | {rec['count']} | {rec['total_us']:,.1f} | "
            f"{rec['total_us'] / total * 100:5.1f} | {rec['total_us'] / rec['count']:,.1f} | "
            f"{rec['max_us']:,.1f} | {rec['min_us']:,.1f} | "
            f"{geo['launch__grid_dim_x']}x{geo['launch__grid_dim_y']}x{geo['launch__grid_dim_z']} | "
            f"{geo['launch__block_dim_x']}x{geo['launch__block_dim_y']}x{geo['launch__block_dim_z']} | "
            f"{geo['launch__registers_per_thread']} |")
    lines.append("")
    lines.append("Read this first: the top row is where time goes. If one "
                 "kernel dominates share, profile it in a standalone harness; "
                 "if many small kernels each take <5 µs, the win is fusion, "
                 "not micro-optimization.")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    analysis = Path(args.run_dir) / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    rows, actions = inventory(args.report)
    out = analysis / f"kernel_inventory_{args.tag}.txt"
    out.write_text(render(rows, actions, args.tag))
    print(f"[{args.tag}] {len(rows)} distinct kernels, {len(actions)} launches "
          f"-> analysis/{out.name}")


if __name__ == "__main__":
    main()
