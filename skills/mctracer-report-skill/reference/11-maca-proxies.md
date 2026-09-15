# MACA Proxies — Substitutes for Counter-Dependent Diagnoses

mcTracer cannot see inside a kernel: no stall reasons, no cache hit rates, no
pipe utilization, no spill detection. When a diagnosis needs one of those,
use one of the sanctioned proxies below, and **state the substitution in the
report** rather than presenting an inferred result as a measurement.

The general technique: replace *observation* with *experiment*. Vary the
thing that would fix the suspected problem, re-time, and let the delta speak.

---

## Proxy 1 — The one-knob sweep

**Replaces:** occupancy limits, register pressure (Pattern K), latency-bound
(Pattern E, partly), pipeline bubbles (Pattern M).

**Method:** rebuild the harness with one compile-time or launch-time knob
changed, re-trace, compare `dur`.

| Knob | Varies | If duration drops, the kernel was… |
|---|---|---|
| block size (64/128/256/512) | warps per block, occupancy | occupancy-limited (Pattern J/K) |
| `-maxrregcount=N` / `__launch_bounds__` | register budget | register-limited (Pattern K) |
| tile / unroll factor | ILP, in-flight loads | latency-bound (Pattern E) |
| double-buffer on/off | compute/memory overlap | pipeline-bubbled (Pattern M) |
| FP32 literals fixed (`1.0` → `1.0f`) | FP64 use | unintentional FP64 (Pattern L) |

**How to record it:** one row per configuration in `benchmark.csv`, then
reference the experiment by ID (E1, E2, …) from the report. A sweep that
moves nothing is also a result — it rules out that dimension.

```csv
exp, kernel, tag, block, regs, tile, duration_us, notes
E1, saxpy, base, 256, 6, 1, 86.65, baseline
E1, saxpy, b512, 512, 6, 1, 61.20, occupancy-limited
```

---

## Proxy 2 — Sort-then-rerun (tail effect)

**Replaces:** per-SM active-cycle distribution (upstream Dimension 2's
imbalance signal).

**Method:** if the workload has variable-length inputs, sort the inputs by
length and re-run. If the tail shrinks, the imbalance was real.

```python
# before profiling
lengths = [len(x) for x in batch]
order = sorted(range(len(batch)), key=lambda i: lengths[i])
sorted_batch = [batch[i] for i in order]
```

**Why it works:** sorting makes CTAs running concurrently do roughly equal
work. If the sorted run is materially faster, load imbalance was the
bottleneck and the fix is packed batching or chunking (Pattern B).

---

## Proxy 3 — Roofline with a supplied op count

**Replaces:** `sm__throughput`, `dram__bytes_read.sum.pct_of_peak_sustained_elapsed`,
tensor-core pipe activity (upstream Dimensions 4 and 6).

**Method:** compute the kernel's arithmetic intensity from its definition,
then place it on the C500 roofline using `roofline()`.

```python
from mctracer_utils import roofline, C500_DEVICE

# 4096² fp32 matmul: 2N³ FLOPs, N² * 2 * 4 bytes moved
N = 4096
flops = 2 * N**3
bytes_moved = 2 * N * N * 4
r = roofline(action, flops=flops)
ai = flops / bytes_moved          # FLOPs / byte
```

- **High arithmetic intensity + achieved FLOPs near peak** → compute-bound.
- **Low arithmetic intensity + slow** → memory-bound; the fix is fewer bytes
  or more reuse.
- **Low achieved FLOPs *and* low bytes** → latency- or launch-bound; check
  Dimension 1 and 2.

**Caveat:** `C500_DEVICE["peak_f32_flops"]` is an estimate. Relative
comparisons between candidates on the same board are sound; absolute
percentages of peak are approximate and should be labeled as such.

---

## Proxy 4 — Source inspection

**Replaces:** per-PC stall hotspots, `pc_to_source_line`.

**Method:** read the harness source. mcTracer cannot attribute time to a
line, but the kernel is usually small enough that the suspect is visible:

- A strided global load (`x[lane * K + i]`) → Pattern C.
- `if (lane_id < K) { out[...] = ... }` → Pattern D.
- `atomicAdd` in a hot loop → Pattern G.
- A `__syncthreads()` inside a tiled loop → Pattern I.
- `1.0` / `0.5` literals in a float kernel → Pattern L.

Then confirm with a one-knob experiment (Proxy 1). The pairing — source says
*suspect*, experiment says *confirmed* — is the MACA equivalent of ncu's
stall hotspot table.

---

## Proxy 5 — Library-kernel comparison

**Replaces:** tensor-core utilization (upstream Dimension 4).

**Method:** profile the reference library implementation of the same
operation and compare.

```python
# The trace of a torch matmul shows the library GEMM:
# mcblas__Mck_tf32gemm_nn_256x256x32_2m1n2k_512t_fp32_fp32_tf3_sb_0_0
```

If the library kernel is N× faster than yours at the same shape, the gap is
in tile shape, tensor-core use, or memory staging — not in anything the trace
measures directly. Use the ratio as the optimization target and the library
kernel's grid/block/regs as a reference configuration.

---

## Proxy 6 — torch profiler for dispatch context

**Replaces (partly):** the in-context view of which PyTorch ops map to which
kernels.

**Method:** when the kernel is reached through PyTorch, torch's own profiler
works on MACA (verified) and names the op → kernel mapping:

```python
from torch.profiler import profile, ProfilerActivity
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    ...
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
prof.export_chrome_trace("/tmp/torch_trace.json")   # same format as mcTracer
```

This is complementary, not equivalent: torch gives op-level attribution,
mcTracer gives per-launch hardware metadata (grid/regs/occupancy). Use both
when the dispatch path is unclear.

---

## What NOT to do

- **Don't fabricate a counter-style number.** "L1 hit rate 92%" cannot exist
  here. Write "not measurable on MACA; inferred from experiment E3" instead.
- **Don't report `queue_ts → complete_ts` as kernel duration.** Use `dur`.
  The two can differ by ~10× on short kernels.
- **Don't present derived occupancy as measured.** `launch_geometry()` is
  arithmetic on device constants; when it disagrees with the runtime's
  `mtreg_occupancy(%)`, the runtime wins and the gap goes in the report.
- **Don't skip the gapped dimensions silently.** A reader comparing this
  report to an NVIDIA-side report needs to know which columns are missing
  and why.
