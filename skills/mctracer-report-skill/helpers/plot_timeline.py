#!/usr/bin/env python3
"""ASCII timeline plots from a mcTracer trace.

Port of the upstream ncu-report-skill `plot_timeline.py`. Upstream plots
PM-sampled SM/DRAM utilization; MACA has no PM sampling, so this plots the
per-launch device timeline instead. Same diagnostic goals:

  - see whether kernels overlap or serialize
  - see queue depth (how many launches are in flight)
  - see tail effects (a few long launches dragging out a batch)
  - see pipeline bubbles (gaps between compute and memory phases)

Usage:
    python3 plot_timeline.py --run-dir $PROFILE_RUN_DIR \
        --report reports/full_v1.json --tag v1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mctracer_utils import action_name, load_report  # noqa: E402

WIDTH = 78  # characters


def timeline_rows(actions):
    """(name, start_us, end_us) per kernel launch, time-ordered."""
    rows = []
    for a in actions:
        ts = a.get("ts", 0) or 0
        dur = a.get("dur", 0) or 0
        rows.append((action_name(a) or "?", ts / 1000.0, (ts + dur) / 1000.0))
    rows.sort(key=lambda r: r[1])
    return rows


def _bar(start, end, t0, span, width=WIDTH):
    lo = int((start - t0) / span * width) if span else 0
    hi = max(lo + 1, int((end - t0) / span * width) if span else 1)
    lo, hi = max(0, lo), min(width, hi)
    return " " * lo + "#" * (hi - lo) + " " * (width - hi)


def render(rows, tag, max_rows=40):
    if not rows:
        return f"# timeline — {tag}\n\n(no kernel launches found)\n"
    t0 = rows[0][1]
    t_end = max(r[2] for r in rows)
    span = t_end - t0 or 1.0
    lines = [f"# timeline — {tag}", "",
             f"span: {span:,.1f} µs   launches: {len(rows)}   "
             f"device time: {sum(r[2] - r[1] for r in rows):,.1f} µs", ""]
    # timestamps in the trace are absolute ns epochs; show them relative to
    # the first launch so the axis labels stay readable.
    shown = rows[:max_rows]
    for name, start, end in shown:
        label = name if len(name) <= 30 else name[:27] + "..."
        lines.append(f"{label:>30} |{_bar(start, end, t0, span)}| {end - start:>9.2f} µs")
    if len(rows) > max_rows:
        lines.append(f"... {len(rows) - max_rows} more launches not shown")
    lines.append("")
    lines.append(f"{'':>30} +{'-' * WIDTH}+")
    lines.append(f"{'':>30} {'0µs':>7}{'':>{WIDTH - 14}}{span:>6.1f}µs")

    lines += ["", "## in-flight launches over time (queue depth)", ""]
    # sample at 60 points; count launches active at each point
    samples = []
    for i in range(60):
        t = t0 + span * i / 59
        n = sum(1 for _, s, e in rows if s <= t < e)
        samples.append(n)
    peak = max(samples) or 1
    for i, n in enumerate(samples):
        bar = "#" * int(n / peak * (WIDTH - 12))
        rel = span * i / 59
        lines.append(f"  t+{rel:>8.1f}µs  {bar} {n}")
    lines.append("")
    lines.append("Interpretation: a flat-high depth means good overlap; a "
                 "single tall spike means one launch at a time (serialized); "
                 "a long tail with depth 1 means a few launches drag out the end.")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--report", action="append", required=True)
    ap.add_argument("--tag", action="append", required=True)
    args = ap.parse_args()

    if len(args.report) != len(args.tag):
        sys.exit("--report and --tag must be given the same number of times")

    analysis = Path(args.run_dir) / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)

    for rep, tag in zip(args.report, args.tag):
        _, actions = load_report(rep)
        txt = render(timeline_rows(actions), tag)
        out = analysis / f"pm_timeline_plots_{tag}.txt"
        out.write_text(txt)
        print(f"[{tag}] -> analysis/{out.name}")


if __name__ == "__main__":
    main()
