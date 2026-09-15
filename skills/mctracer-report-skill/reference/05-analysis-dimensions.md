# Analysis Dimensions

Every kernel profile report is ambiguous until you look at it through
specific lenses. Upstream KDA defines six; on MACA four are fully supported
by trace data and two are gapped. Walk through all of them; don't stop at the
first finding, and say plainly which ones you could not run.

For each dimension: what you're answering, which fields to read, how to read
them, and which helper to run.

---

## Dimension 1 — SM occupancy & launch geometry  ✅ supported

**What:** is the grid large enough to fill the 104 SMs? Is occupancy being
limited by registers, shared memory, or block-size constraints?

**Fields (derived — see `launch_geometry()`):**
```
launch__grid_size
launch__block_size
launch__grid_dim_x / _y / _z
launch__waves_per_multiprocessor
launch__registers_per_thread
launch__shared_mem_per_block
launch__occupancy_limit_blocks
launch__occupancy_limit_registers
launch__occupancy_limit_shared_mem
launch__occupancy_limit_warps
device__attribute_multiprocessor_count      (104 on C500)
sm__maximum_warps_per_active_cycle_pct      (theoretical occupancy %)
```

**Fields (reported by the MACA runtime):**
```
args["mtreg_occupancy(%)"]              # register-limited occupancy
args["shared_memeory_occupancy(%)"]     # shared-memory-limited occupancy
```

**Reading:**

- **Waves / SM < 1**: grid is too small to fill the chip. With 104 SMs, if
  `launch__grid_size < 104 × blocks_per_SM`, some SMs sit idle the entire
  time. This is by far the most common finding on MACA traces, and the
  highest-leverage fix.
- **Waves / SM in [1, 2)**: you have a tail wave (partial last wave). Tail
  cost is roughly `(last_wave_blocks / wave_size) × (block_exec_time /
  total_kernel_time)` — `launch_geometry()` computes both factors.
- **Waves / SM > 4**: grid is plenty big, scheduling averages out.
- **Theoretical occupancy 100% but the runtime reports low**: on MACA you
  cannot see achieved occupancy (no `sm__warps_active`), so reason from the
  limits: whichever of registers/shared-mem/warps binds first is the lever.
- **Theoretical occupancy < 100% and `launch__occupancy_limit_registers` is
  the tightest**: reduce register usage or add `__launch_bounds__`.
- **`launch__occupancy_limit_shared_mem` the tightest**: shared mem / block
  is too large, reduce tile size.

**Derived: wave math** (identical to upstream formula)

```python
blocks_per_sm = min(occ_limit_blocks, occ_limit_registers, occ_limit_shared_mem, occ_limit_warps)
wave_size = blocks_per_sm * num_sms
num_waves = (total_blocks + wave_size - 1) // wave_size
last_wave_blocks = total_blocks - (num_waves - 1) * wave_size
last_wave_utilization_pct = last_wave_blocks / wave_size * 100
```

**Helper:** `analyze_reports.py` prints all key launch metrics in
`metrics_key_<tag>.txt`.

---

## Dimension 2 — Per-launch timeline: overlap & tail  ✅ supported

**What:** do kernels serialize behind each other, or overlap? Does a handful
of slow launches drag out the whole run?

**Fields:**
```
ts                # device-side start, ns
dur               # device-side duration, ns
queue_ts          # host enqueue start
submit_ts         # host enqueue end
```

**Reading the in-flight shape** (from `plot_timeline.py`'s depth plot):

- **Flat high → clean drop**: ideal. Launches overlap, steady state.
- **Flat high → gradual tail**: tail effect. The tail's length is how much
  time a few slow launches waste. Usually caused by variable-length inputs
  (seq_len varies per batch element).
- **Depth 1 throughout**: fully serialized. Either the launches are giant, or
  host-side work dominates the gaps. Check `submit_ts - queue_ts` for the
  host-side cost.
- **Periodic gaps**: host-side work between launches — allocation, Python
  overhead, framework bookkeeping. The fusion opportunity.

**Where tail typically comes from:**

1. **Variable-length per-CTA work**: each CTA's iteration count depends on an
   input axis (per-element lengths driven by a prefix-sum array), so CTAs
   take very different times.
2. **Branch-and-early-exit inside the kernel**: some blocks bail early via
   `return`, others don't.
3. **Work-stealing without load balancing**: custom scheduling that happens
   to assign heavy work to a few blocks.

**Fix direction:** chunk the variable-length work (time-chunking for
sequence-style workloads), or oversubscribe with work-stealing.

**Helper:** `plot_timeline.py` → `pm_timeline_plots_<tag>.txt`.

**Caveat vs upstream:** ncu's PM sampling showed utilization *within* a
kernel. This dimension shows timing *between* launches. Intra-kernel tail
(imbalance across CTAs of one launch) is not directly observable on MACA —
reason from the input distribution instead (see `11-maca-proxies.md`).

---

## Dimension 3 — Kernel inventory: where time goes  ✅ supported

**What:** which kernels actually consume the run's time?

**Fields:** per-kernel aggregates over the trace (name, count, total/max/min
duration).

**Reading:**

- **One kernel at >70% share**: profile it in a standalone harness; the rest
  is noise.
- **Many kernels each <5 µs**: the win is kernel fusion, not micro-
  optimization of any single one. Every small launch pays fixed overhead.
- **A framework kernel dominating** (`mcblas__...`, `mcGetDeviceProperties`):
  the harness is doing more setup than work — build a tighter harness before
  concluding anything about your kernel.
- **High launch count for the same kernel**: batch the launches.

**Helper:** `kernel_inventory.py` → `kernel_inventory_<tag>.txt`. This view
has no upstream equivalent — ncu's replay model only ever profiled one kernel
per report.

---

## Dimension 4 — Roofline position  ✅ supported (with input)

**What:** is the kernel compute-bound or memory-bound?

**This is the one dimension where you must supply external knowledge**: the
kernel's arithmetic op count. mcTracer cannot count instructions.

```python
from mctracer_utils import roofline

# For a 4096x4096 fp32 matmul: 2 * 4096^3 FLOPs
r = roofline(actions[0], flops=2 * 4096**3)
# {"duration_s": ..., "achieved_flops": ..., "achieved_flops_pct_of_peak": ...}
```

**Reading:**

- **achieved FLOPs near the peak**: compute-bound. The remaining wins are
  algorithmic (fewer ops) or tensor-core-shaped.
- **achieved FLOPs far below peak and the kernel moves lots of bytes**:
  memory-bound. Reduce bytes moved / increase reuse.
- **Neither**: likely latency-bound or launch-bound — check Dimension 1 and 2.

**Caveat:** `C500_DEVICE["peak_f32_flops"]` is an estimate (104 SM × 2048
threads × 32 FMA/cycle × derived clock). Verify against the board's vendor
spec before quoting absolute percentages; relative comparisons between two
candidates on the same board are sound regardless.

---

## Dimension 5 — Stall reason breakdown  ❌ gapped on MACA

**What (upstream):** when warps aren't issuing, what are they waiting for?
Which source lines generate the most stalls?

**Not available.** mcTracer records no warp-state sampling, no per-PC stall
samples, and has no `smsp__pcsamp_*` equivalent. There is no field to read.

**Do not** write around stall reasons as if you had measured them. Use
[`11-maca-proxies.md`](11-maca-proxies.md) instead: a controlled timing
experiment (vary block size, tile size, unroll factor, shared-memory usage)
and infer the binding constraint from which knob moves the number.

**What you can still say honestly:** "the kernel is X µs with grid G and
occupancy O; MACA does not sample stall reasons, so the binding constraint
was identified by varying the tile factor (see experiment E2)."

---

## Dimension 6 — Memory access pattern & cache efficiency  ❌ gapped on MACA

**What (upstream):** are global loads coalesced? Are caches hit? Is DRAM
actually busy? (`dram__bytes_read.sum.pct_of_peak_sustained_elapsed`,
`l1tex__t_sector_hit_rate.pct`, sectors/request, store efficiency, register
spill via `local_ld`.)

**Not available.** mcTracer reports `bytes` only for explicit device copies
(`mcMemcpy`), not for a kernel's own load/store traffic. No cache counters,
no sector counts.

**Do not** report "DRAM throughput" or "L1 hit rate" — those numbers do not
exist on this platform. Use the roofline estimate (Dimension 4) as the
closest honest substitute, plus static reasoning about the access pattern
from the source.

---

## Cross-dimension synthesis

After walking through all six (including the two you could not run), write a
one-line diagnosis. Name the signals you have, each tied to a dimension:

> "The kernel runs 86 µs with a 4096-block grid on 104 SMs
> (`launch__waves_per_multiprocessor = 1.23`, Dim 1), leaving the last wave
> 23% utilized. Register-limited occupancy is 1% with 6 regs/thread and
> 256-thread blocks (Dim 1). The run is otherwise serialized — in-flight
> depth stays at 1 (Dim 2). MACA does not sample stall reasons or cache
> efficiency (Dim 5, 6), so the binding constraint was identified by the
> tile-factor sweep in experiment E2."

Fill in the values from your own trace, and state the gaps rather than
papering over them. That sentence is the deliverable — everything else in the
report is evidence backing it.
