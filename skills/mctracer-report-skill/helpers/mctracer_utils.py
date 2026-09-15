"""Shared helpers for parsing MACA mcTracer reports.

The upstream KDA ships ncu_utils.py built on NVIDIA's `ncu_report` module.
MACA has no Nsight Compute; the equivalent evidence source is the Chrome
trace-event JSON emitted by `mcTracer`. This module mirrors the ncu_utils
public surface (load_report / safe / per_instance / curated metric sets)
so the ported skill keeps the same shape as upstream.

Usage:
    from mctracer_utils import load_report, safe, C500_KEY_METRICS

One "action" == one kernel launch. The kernel-launch X-event carries its
launch geometry, register count and occupancies in `args`; `dur` is the
device-side execution time in nanoseconds.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

# --- Loading -----------------------------------------------------------------

def load_report(path):
    """Load a mcTracer JSON trace and return (trace, actions).

    An action is the kernel-launch X-event. Returns (trace, [action, ...])
    to mirror ncu_utils' (report, action) tuple, but a trace usually holds
    many launches, so all profiled kernels are returned.
    """
    trace = json.loads(Path(path).read_text())
    actions = [e for e in trace["traceEvents"]
               if e.get("ph") == "X" and "grid" in e.get("args", {})]
    return trace, actions


def load_action(path, name_regex=None):
    """Return the first (or first matching) kernel-launch action."""
    _, actions = load_report(path)
    if name_regex is None:
        return actions[0] if actions else None
    import re
    rx = re.compile(name_regex)
    for a in actions:
        if rx.search(action_name(a) or ""):
            return a
    return None


def action_name(action):
    """Demangled-ish name. mcTracer gives a mangled `_Z12saxpy...` in args
    and the pretty `saxpy_kernel(int, ...)` on the event itself."""
    return action.get("name") or action.get("args", {}).get("name", "?")


# --- Safe metric access ------------------------------------------------------

def safe(action, name, default=None):
    """Return metric value, or `default` if the metric is missing.

    Accepts the MACA-native field name (`grid`, `registers_per_thread`), a
    dotted args path (`mem.registers_per_thread`), or an ncu-semantically-
    equivalent name (`launch__grid_size`) resolved through METRIC_ALIASES.
    """
    try:
        if name in action:
            return action[name]
        args = action.get("args", {})
        if name in args:
            return args[name]
        if "." in name:
            v = _lookup(args, name)
            if v is not None:
                return v
        alias = METRIC_ALIASES.get(name)
        if alias == "shared_mem_total":
            mem = args.get("mem", {})
            st, dyn = mem.get("static_shared"), mem.get("dynamic_shared")
            if st is None and dyn is None:
                return default
            return (st or 0) + (dyn or 0)
        if alias:
            return _lookup(args, alias)
        return default
    except Exception:
        return default


def _lookup(args, dotted):
    """Resolve 'a.b.c' style nested keys inside the event args."""
    cur = args
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def safe_many(action, names, default=None):
    """Bulk-fetch multiple metrics. Returns a dict name -> value-or-default."""
    return {n: safe(action, n, default) for n in names}


# --- Derived launch geometry (upstream wave math) -----------------------------

def launch_geometry(action, device=None):
    """Recompute the upstream `launch__*` family from raw fields.

    ncu exposes waves_per_multiprocessor and occupancy limits directly;
    on MACA we derive them from grid/block/regs and the device properties.
    """
    grid = safe(action, "launch__grid_dim_x", 1) or 1
    gy, gz = (safe(action, "launch__grid_dim_y", 1) or 1,
              safe(action, "launch__grid_dim_z", 1) or 1)
    block = safe(action, "launch__block_dim_x", 1) or 1
    by, bz = (safe(action, "launch__block_dim_y", 1) or 1,
              safe(action, "launch__block_dim_z", 1) or 1)
    regs = safe(action, "launch__registers_per_thread", 0) or 0

    total_blocks = grid * gy * gz
    threads_per_block = block * by * bz
    if device is None:
        device = C500_DEVICE
    regs_per_sm = device["regs_per_multiprocessor"]
    max_threads_per_sm = device["max_threads_per_multiprocessor"]
    max_blocks_per_sm = device.get("max_blocks_per_multiprocessor", 32)
    smem_per_sm = device["shared_memory_per_multiprocessor"]

    blocks_by_regs = (regs_per_sm // (regs * threads_per_block)) if regs and threads_per_block else max_blocks_per_sm
    blocks_by_threads = max_threads_per_sm // threads_per_block if threads_per_block else 0
    smem = safe(action, "launch__shared_mem_per_block", 0) or 0
    blocks_by_smem = (smem_per_sm // smem) if smem else max_blocks_per_sm
    blocks_per_sm = max(1, min(max_blocks_per_sm, blocks_by_regs, blocks_by_threads, blocks_by_smem))

    wave_size = blocks_per_sm * device["multiprocessor_count"]
    num_waves = (total_blocks + wave_size - 1) // wave_size if wave_size else 0
    last_wave_blocks = total_blocks - (num_waves - 1) * wave_size if num_waves else 0
    return {
        "launch__grid_size": total_blocks,
        "launch__block_size": threads_per_block,
        "launch__grid_dim_x": grid, "launch__grid_dim_y": gy, "launch__grid_dim_z": gz,
        "launch__block_dim_x": block, "launch__block_dim_y": by, "launch__block_dim_z": bz,
        "launch__thread_count": total_blocks * threads_per_block,
        "launch__registers_per_thread": regs,
        "launch__shared_mem_per_block": smem,
        "launch__waves_per_multiprocessor": (total_blocks / wave_size) if wave_size else 0,
        "launch__occupancy_limit_blocks": max_blocks_per_sm,
        "launch__occupancy_limit_registers": blocks_by_regs,
        "launch__occupancy_limit_shared_mem": blocks_by_smem,
        "launch__occupancy_limit_warps": blocks_by_threads,
        "wave_size": wave_size,
        "num_waves": num_waves,
        "last_wave_blocks": last_wave_blocks,
        "last_wave_utilization_pct": (last_wave_blocks / wave_size * 100) if wave_size else 0,
    }


def theoretical_occupancy_pct(action, device=None):
    """sm__maximum_warps_per_active_cycle_pct equivalent."""
    geo = launch_geometry(action, device)
    return min(100.0, geo["launch__occupancy_limit_registers"] *
               geo["launch__block_size"] / (device or C500_DEVICE)["max_threads_per_multiprocessor"] * 100)


# --- Occupancy-reported vs measured -------------------------------------------

def reported_occupancy(action):
    """mcTracer reports register- and shared-memory-limited occupancy directly.

    Returns dict with the two percentages, or None when absent.
    """
    args = action.get("args", {})
    mtreg = args.get("mtreg_occupancy(%)")
    smem = args.get("shared_memeory_occupancy(%)")
    if mtreg is None and smem is None:
        return None
    return {"occupancy_limit_registers_pct": mtreg,
            "occupancy_limit_shared_mem_pct": smem}


# --- Per-instance (per-launch / per-timestamp) values -------------------------

def per_launch_values(trace, metric):
    """All per-launch values of a metric across the trace (ncu per-instance)."""
    _, actions = load_report(trace) if isinstance(trace, (str, Path)) else (trace, trace["actions"])
    return [safe(a, metric) for a in actions]


def kernel_timeline(trace, name_regex=None):
    """Time-ordered (ts, dur, name) for every kernel launch.

    This is the MACA substitute for the ncu PM-sampling timeline: mcTracer
    does not sample SM/DRAM utilization over time, but it does give exact
    per-launch start/end, which is enough to see queue depth and overlap.
    """
    import re
    _, actions = load_report(trace) if isinstance(trace, (str, Path)) else (trace, trace["actions"])
    rx = re.compile(name_regex) if name_regex else None
    out = []
    for a in actions:
        nm = action_name(a)
        if rx and not rx.search(nm or ""):
            continue
        out.append((a.get("ts", 0), a.get("dur", 0), nm))
    out.sort(key=lambda t: t[0])
    return out


# --- Roofline / achieved throughput ------------------------------------------

def roofline(action, flops, device=None):
    """Rough roofline position. `flops` = arithmetic ops of the kernel.

    mcTracer has no FMA/LSU pipe utilization counters, so the compute side
    is estimated from the kernel's known op count and measured duration.
    """
    if device is None:
        device = C500_DEVICE
    dur_s = (safe(action, "gpu__time_duration.sum", 0) or 0) / 1e9
    if not dur_s:
        return None
    return {
        "duration_s": dur_s,
        "achieved_flops": flops / dur_s if flops else None,
        "achieved_flops_pct_of_peak": (flops / dur_s / device["peak_f32_flops"]) if flops else None,
    }


# --- Archive all metrics -----------------------------------------------------

def dump_all_metrics(action, outfile):
    """Dump every field of a kernel-launch action to JSON (ncu dump_all_metrics)."""
    args = dict(action.get("args", {}))
    args["__event_name"] = action.get("name")
    args["__ts"] = action.get("ts")
    args["__dur"] = action.get("dur")
    args["__cat"] = action.get("cat")
    Path(outfile).write_text(json.dumps(args, indent=1, default=str))
    return len(args)


# --- Curated metric sets -----------------------------------------------------
#
# Names follow the upstream B200_KEY_METRICS convention where MACA provides a
# semantically equivalent value, so downstream analysis text stays portable.
# Fields MACA cannot observe (stall reasons, cache hit rates, tensor-core
# pipe activity) are deliberately absent — see reference/10-maca-mapping.md.

C500_DEVICE = {
    "name": "MetaX C500",
    "multiprocessor_count": 104,
    "max_threads_per_multiprocessor": 2048,
    "regs_per_multiprocessor": 131072,
    "shared_memory_per_block": 65536,
    "shared_memory_per_multiprocessor": 65536,
    "warp_size": 64,
    "l2_cache_size": 8388608,
    "total_memory": 16341008384,
    "max_blocks_per_multiprocessor": 32,
    "peak_f32_flops": 104 * 2048 * 32 * 1.6e9,  # 104 SM x 2048 thr x 32 FMA/cycle, clock-derived estimate
    "peak_memory_bandwidth": 3.2e12,            # bytes/s, vendor-declared HBM peak (verify per board)
}

# ncu metric name -> dotted path into the mcTracer event args.
METRIC_ALIASES = {
    "launch__grid_dim_x": "grid.x",
    "launch__grid_dim_y": "grid.y",
    "launch__grid_dim_z": "grid.z",
    "launch__block_dim_x": "block.x",
    "launch__block_dim_y": "block.y",
    "launch__block_dim_z": "block.z",
    "launch__registers_per_thread": "mem.registers_per_thread",
    "launch__shared_mem_per_block_static": "mem.static_shared",
    "launch__shared_mem_per_block_dynamic": "mem.dynamic_shared",
    "gpu__time_duration.sum": "dur",          # ns, device-side
    "launch__shared_mem_per_block": "shared_mem_total",  # computed in safe()
}

C500_KEY_METRICS = [
    # Launch geometry (derived — see launch_geometry)
    "launch__grid_size",
    "launch__block_size",
    "launch__grid_dim_x", "launch__grid_dim_y", "launch__grid_dim_z",
    "launch__block_dim_x", "launch__block_dim_y", "launch__block_dim_z",
    "launch__waves_per_multiprocessor",
    "launch__registers_per_thread",
    "launch__shared_mem_per_block",
    "launch__occupancy_limit_registers",
    "launch__occupancy_limit_shared_mem",
    "device__attribute_multiprocessor_count",
    # Timing
    "gpu__time_duration.sum",
    # Occupancy (reported by the MACA runtime)
    "occupancy_limit_registers_pct",
    "occupancy_limit_shared_mem_pct",
    # Memory volume (only when the kernel issues device-visible copies)
    "bytes",
]
