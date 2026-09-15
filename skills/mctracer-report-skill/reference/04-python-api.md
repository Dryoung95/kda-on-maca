# `mctracer_utils` Python API

Use this module, not raw JSON, for anything beyond a quick look. It mirrors
the upstream `ncu_report` / `ncu_utils.py` public surface so analysis scripts
stay portable across the two backends. Pure stdlib — no `PYTHONPATH` setup,
no CUDA install to locate.

```python
import sys; sys.path.insert(0, "skills/mctracer-report-skill/helpers")
from mctracer_utils import load_report, safe, action_name
```

---

## Basic loading

```python
from mctracer_utils import load_report

trace, actions = load_report("reports/<tag>-12345.json")
# actions = every kernel-launch X-event, in trace order

print(f"Launches: {len(actions)}")
print(f"First kernel: {action_name(actions[0])}")
```

One action = one kernel launch. `load_action(path, name_regex)` returns just
the first (or first matching) launch — the direct analogue of ncu's
`load_action`.

---

## Reading a single metric

```python
from mctracer_utils import safe

a = actions[0]
dur_ns = safe(a, "gpu__time_duration.sum")        # ns, device-side
grid_x = safe(a, "launch__grid_dim_x")
regs = safe(a, "launch__registers_per_thread")

print(f"Duration: {dur_ns/1e3:.2f} µs")
print(f"Grid x:   {grid_x}")
print(f"Regs:     {regs}")
```

**Always wrap in `safe()`.** Two name families are accepted:

1. **ncu-semantically-equivalent names** — `"launch__grid_dim_x"`,
   `"launch__registers_per_thread"`, `"gpu__time_duration.sum"`. Resolved
   through `METRIC_ALIASES` to the underlying mcTracer field. Use these in
   analysis prose so text stays portable to the NVIDIA backend.
2. **MACA-native field paths** — `"grid.x"`, `"mem.registers_per_thread"`,
   `"dur"`. Direct dotted lookups into the event args. Use these when the
   ncu analogue does not exist.

A name that is neither returns `None` (never raises).

---

## Enumerating available metrics

```python
from mctracer_utils import dump_all_metrics

# Every field of one launch, archived to JSON
n = dump_all_metrics(actions[0], "analysis/metrics_all_v1_0.json")
print(f"wrote {n} fields")
```

To discover field names interactively, pretty-print one action's args:

```python
import json
print(json.dumps(actions[0]["args"], indent=1))
```

This is the analogue of ncu's `action.metric_names()` — much smaller (a dozen
fields vs 2000+ metrics) because mcTracer records launch metadata, not
hardware counters.

---

## Derived launch geometry (the upstream `launch__*` family)

ncu exposes `launch__waves_per_multiprocessor` and the `launch__occupancy_*`
limits directly. mcTracer does not, so the module derives them from
grid/block/registers and the C500 device properties:

```python
from mctracer_utils import launch_geometry, C500_DEVICE

geo = launch_geometry(actions[0], device=C500_DEVICE)
# {
#   "launch__grid_size": 4096, "launch__block_size": 256,
#   "launch__grid_dim_x": 4096, ..., "launch__block_dim_x": 256, ...,
#   "launch__thread_count": 1048576,
#   "launch__registers_per_thread": 6,
#   "launch__shared_mem_per_block": 0,
#   "launch__waves_per_multiprocessor": 1.23,
#   "launch__occupancy_limit_blocks": 32,
#   "launch__occupancy_limit_registers": 85,
#   "launch__occupancy_limit_shared_mem": 32,
#   "launch__occupancy_limit_warps": 8,
#   "wave_size": 3328, "num_waves": 2,
#   "last_wave_blocks": 768, "last_wave_utilization_pct": 23.08,
# }
```

The wave math is the upstream formula, verbatim:

```python
blocks_per_sm = min(occ_limit_blocks, occ_limit_registers,
                    occ_limit_shared_mem, occ_limit_warps)
wave_size = blocks_per_sm * num_sms
num_waves = (total_blocks + wave_size - 1) // wave_size
last_wave_blocks = total_blocks - (num_waves - 1) * wave_size
last_wave_utilization_pct = last_wave_blocks / wave_size * 100
```

**Caveat:** `launch__occupancy_limit_registers` assumes 131072 registers per
SM and uniform register allocation — the same assumption ncu makes. If the
runtime-reported `mtreg_occupancy(%)` disagrees with the derived value, the
runtime number wins; file the discrepancy in the report.

---

## Per-launch timeline (the PM-sampling substitute)

```python
from mctracer_utils import kernel_timeline

rows = kernel_timeline("reports/<tag>-12345.json")
# [(start_ns, dur_ns, name), ...] sorted by start
```

Upstream's PM sampling gave SM/DRAM utilization *over time within a kernel*.
This is per-launch timing across the run — a coarser but honest signal: it
shows serialization, queue depth, and tail, not intra-kernel utilization.

Shape reading (see also `plot_timeline.py`):

- **Flat-high in-flight count**: launches overlap — good throughput.
- **Depth 1 throughout**: fully serialized; host-side gaps dominate.
- **Long tail of depth 1 at the end**: a few slow launches drag out the run.
- **Gaps between launches**: host-side work (alloc, Python overhead) between
  kernels — the fusion opportunity.

---

## Kernel inventory helper

```python
from kernel_inventory import inventory

rows, all_actions = inventory("reports/<tag>-12345.json")
for name, rec in sorted(rows, key=lambda r: -r[1]["total_us"])[:10]:
    print(f"{rec['total_us']:10.1f} µs  {rec['count']:5d}x  {name}")
```

This view has no upstream equivalent — ncu profiled one kernel at a time and
never saw the whole run. On MACA it is the cheapest first look: time share by
kernel, before any per-kernel analysis.

---

## What is NOT available (read this before writing a diagnosis)

These `ncu_utils` functions have **no MACA equivalent** and must not be
substituted with invented numbers:

| Upstream | Status on MACA |
|---|---|
| `per_pc_values(action, "smsp__pcsamp_*")` | ✗ no per-PC sampling |
| `pc_to_source_line(action, pc)` | ✗ no PC → source mapping |
| `rule_results(action)` / `rule_speedups(action)` | ✗ no rule engine |
| `sm__pipe_tensor_cycles_active.*` | ✗ no pipe utilization |
| `dram__bytes_read.sum` / hit rates | ✗ (only explicit copy bytes) |
| `smsp__average_warps_issue_stalled_*` | ✗ no stall sampling |

For each, see [`11-maca-proxies.md`](11-maca-proxies.md) for the sanctioned
fallback (usually a timing experiment or a static reasoning step), and state
the substitution in the report rather than hiding it.
