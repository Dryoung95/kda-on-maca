#!/usr/bin/env python3
"""Extract key metrics from mcTracer JSON traces and compare them.

Port of the upstream ncu-report-skill `analyze_reports.py`. Same CLI shape
(--run-dir / --report / --tag, multiple reports side-by-side), MACA backend.

Usage:
    python3 analyze_reports.py --run-dir $PROFILE_RUN_DIR \
        --report reports/full_v1.json --tag v1 \
        --report reports/full_v2.json --tag v2

Writes metrics_key_<tag>.{txt,json} and (with 2+ reports)
compare_<a>_vs_<b>.txt under <run-dir>/analysis/.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mctracer_utils import (  # noqa: E402
    C500_DEVICE, C500_KEY_METRICS, action_name, dump_all_metrics, launch_geometry,
    load_report, reported_occupancy, safe,
)


def collect_metrics(report_path):
    """Return one metrics dict per kernel launch in the trace."""
    _, actions = load_report(report_path)
    out = []
    for a in actions:
        geo = launch_geometry(a)
        occ = reported_occupancy(a) or {}
        m = {n: safe(a, n) for n in C500_KEY_METRICS}
        m.update(geo)
        m.update(occ)
        m["sm__maximum_warps_per_active_cycle_pct"] = _theoretical_occupancy(geo)
        m["device__attribute_multiprocessor_count"] = C500_DEVICE["multiprocessor_count"]
        m["__name"] = action_name(a)
        m["__mangled"] = a.get("args", {}).get("name")
        m["__dur_us"] = (a.get("dur", 0) or 0) / 1000.0
        out.append(m)
    return out


def _theoretical_occupancy(geo):
    threads = geo["launch__block_size"]
    if not threads:
        return None
    return min(100.0, geo["launch__occupancy_limit_registers"] * threads
               / C500_DEVICE["max_threads_per_multiprocessor"] * 100)


def fmt(v, unit=""):
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:,.2f}{unit}"
    if isinstance(v, int):
        return f"{v:,}{unit}"
    return str(v)


KEY_ROWS = [
    ("Duration", "__dur_us", "us"),
    ("Grid size", "launch__grid_size", ""),
    ("Block size (threads)", "launch__block_size", ""),
    ("Grid dim (x,y,z)", None, ""),
    ("Waves / SM", "launch__waves_per_multiprocessor", ""),
    ("Registers / thread", "launch__registers_per_thread", ""),
    ("Shared mem / block (B)", "launch__shared_mem_per_block", ""),
    ("Theoretical occupancy %", "sm__maximum_warps_per_active_cycle_pct", ""),
    ("Occupancy limit: registers", "launch__occupancy_limit_registers", " blocks/SM"),
    ("Occupancy limit: shared mem", "launch__occupancy_limit_shared_mem", " blocks/SM"),
    ("Reg-limited occupancy % (runtime)", "occupancy_limit_registers_pct", ""),
    ("Smem-limited occupancy % (runtime)", "occupancy_limit_shared_mem_pct", ""),
]


def render_table(launches, tag):
    lines = [f"# Key metrics — {tag}", ""]
    for idx, m in enumerate(launches):
        title = m["__name"] or "<unknown>"
        lines.append(f"## launch {idx}: {title}")
        lines.append("")
        lines.append("| Metric | Value | Source |")
        lines.append("|---|---:|---|")
        for label, key, unit in KEY_ROWS:
            if key is None:
                gx, gy, gz = (m.get(f"launch__grid_dim_{c}") for c in "xyz")
                bx, by, bz = (m.get(f"launch__block_dim_{c}") for c in "xyz")
                lines.append(f"| {label} | ({gx},{gy},{gz}) / ({bx},{by},{bz}) | grid/block args |")
            else:
                src = "derived (launch_geometry)" if key.startswith("launch__") else "trace field"
                lines.append(f"| {label} | {fmt(m.get(key), unit)} | {src} |")
        lines.append("")
    return "\n".join(lines)


def render_compare(launches_by_tag, tags):
    """Side-by-side on the first kernel of each report (same harness, same shape)."""
    lines = ["# Side-by-side comparison", ""]
    header = "| Metric | " + " | ".join(tags) + " |"
    sep = "|---|" + "---:|" * len(tags)
    lines += [header, sep]
    for label, key, unit in KEY_ROWS:
        cells = []
        for t in tags:
            ms = launches_by_tag[t]
            if not ms:
                cells.append("n/a")
                continue
            if key is None:
                m = ms[0]
                gx, gy, gz = (m.get(f"launch__grid_dim_{c}") for c in "xyz")
                cells.append(f"({gx},{gy},{gz})")
            else:
                cells.append(fmt(ms[0].get(key), unit))
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("Compared on the first matching launch of each report. "
                 "If the harnesses differ, compare full metrics_key files instead.")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-dir", required=True, help="profile/<run_name>")
    ap.add_argument("--report", action="append", required=True,
                    help="Path to a mcTracer .json (repeatable)")
    ap.add_argument("--tag", action="append", required=True,
                    help="Label for the preceding --report (repeatable)")
    args = ap.parse_args()

    if len(args.report) != len(args.tag):
        sys.exit("--report and --tag must be given the same number of times")

    run_dir = Path(args.run_dir)
    analysis = run_dir / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)

    by_tag = {}
    for rep, tag in zip(args.report, args.tag):
        launches = collect_metrics(rep)
        by_tag[tag] = launches
        (analysis / f"metrics_key_{tag}.txt").write_text(render_table(launches, tag))
        (analysis / f"metrics_key_{tag}.json").write_text(json.dumps(launches, indent=1, default=str))
        # full archive of every trace field, mirroring ncu's metrics_all_<tag>.json
        _, actions = load_report(rep)
        for i, a in enumerate(actions):
            dump_all_metrics(a, analysis / f"metrics_all_{tag}_{i}.json")
        print(f"[{tag}] {len(launches)} kernel launch(es) -> analysis/metrics_key_{tag}.txt")

    if len(by_tag) >= 2:
        tags = list(by_tag)
        for i in range(len(tags) - 1):
            a, b = tags[i], tags[i + 1]
            (analysis / f"compare_{a}_vs_{b}.txt").write_text(render_compare(by_tag, [a, b]))
            print(f"comparison -> analysis/compare_{a}_vs_{b}.txt")


if __name__ == "__main__":
    main()
