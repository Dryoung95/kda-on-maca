# Diagnosis Playbook — Pattern → Cause → Fix

For each pattern, what does it typically mean, and what's the first fix to
try? Ported from the upstream ncu-report-skill playbook — same pattern
structure, same fixes, with a MACA observability tag on each one:

- ✅ **diagnosable from a mcTracer trace alone**
- 🔶 **needs a proxy** — a controlled timing experiment or external knowledge
- ❌ **not observable on MACA** — must be stated as a gap

Read this after you've gathered the metrics (via
[`05-analysis-dimensions.md`](05-analysis-dimensions.md)) — here you translate
metrics into diagnoses and fix directions.

---

## How to use this doc

For each *observation* below, read:

- **Signals** — what specific values flag this pattern.
- **Why** — the underlying cause.
- **First-line fix** — the cheapest change to try.
- **Deeper fixes** — when first-line isn't enough.
- **Exceptions** — kernel types where this pattern is actually *expected* and
  should be left alone.

Most kernels will match 2-4 patterns simultaneously. **Rank them by
magnitude.** Without ncu's `Est. Speedup` field, rank by: (a) how far the
observed value is from healthy, (b) the share of kernel time the affected
code accounts for. Fix the biggest one first.

---

## Pattern A — Small grid / SM idle  ✅

**Signals:**
- `launch__waves_per_multiprocessor < 0.5`
- `launch__grid_size < 104` (fewer blocks than SMs on C500)
- In the inventory, the kernel takes the majority of run time despite a small grid

**Why:** each CTA occupies at most one SM; with fewer CTAs than SMs, some SMs
are completely idle throughout the kernel.

**First-line fix:** increase grid size. Look for a dimension the kernel
currently doesn't parallelize:
- Add a split along `K` (split-K for reductions / attention).
- Split across heads / channels if grouped.
- Grid-stride loops so one block does multiple work units — but only if work
  units are cheap.

**Deeper fixes:**
- **Persistent kernel**: launch one block per SM, each block dequeues work
  items from an atomic counter. Good for dynamic-shape cases.
- **Fuse with adjacent kernels** so more work fits in one launch.

**Exceptions:**
- LLM decode (batch=1, query_len=1) is fundamentally small. Split-K over KV
  length is the standard mitigation.
- Final reduction stages of a multi-level reduction are naturally small; fuse
  them into the producing kernel.

---

## Pattern B — Tail effect (variable-length inputs)  ✅ (run-level)

**Signals:**
- In-flight depth plot: long gradual tail at the end (visible via
  `plot_timeline.py`).
- `launch__waves_per_multiprocessor > 1.05` with partial last wave
  (`last_wave_utilization_pct` low).
- Input distribution: `max_seq_len / avg_seq_len > 3`.

**Why:** each CTA iterates some variable-size inner loop. When sequences have
vastly different lengths, a few long-sequence CTAs keep running after
everyone else finished.

**First-line fix (cheap):**
- **Packed batching / sorting**: sort inputs by length (at the application
  level) so CTAs running concurrently do roughly equal work.
- **Split long sequences across more CTAs**: add a `split_factor` grid
  dimension; each CTA handles `ceil(seq_len / split_factor)` tokens, and a
  small post-reduction combines partials.

**Deeper fixes:**
- **Chunkwise kernel**: break each sequence into fixed-size chunks, process
  chunks in parallel, then stitch with a small recurrence. This is the
  approach of flash-linear-attention's `chunk_delta_rule_fwd` for
  Mamba/GLA-style recurrences.
- **Classify-and-dispatch**: short sequences go through the simple path (one
  CTA per seq), long sequences through the chunked path.

**Exceptions:**
- Short kernels (< 10 µs) where tail cost is absolute-small.
- Workloads where you already pre-sort / pre-pack.

**MACA caveat:** the trace shows tail across *launches*, not across CTAs
within one launch. For intra-kernel imbalance, reason from the input
distribution and confirm with a sort-then-rerun experiment.

---

## Pattern C — Uncoalesced global loads  ❌

**Signals (upstream):** sectors/request > 5, NCU's "excessive sectors" rule,
`long_scoreboard` on the offending load line.

**MACA status:** not observable. There are no sector counters and no per-PC
stall data.

**What to do instead:** read the source. A strided access (`x[lane * K + i]`)
or AoS layout is a coalescing problem on sight; confirm by fixing it and
re-timing. State the substitution in the report.

**First-line fix:** rework the thread ↔ data mapping:
- If current pattern is `x[lane * K + i]` (stride K), flip to
  `x[lane + i * 32]` (coalesced). (On C500 with warp size 64, the coalesced
  unit is wider — aim for 64-lane-contiguous groups.)
- Check AoS layouts: `struct { float a, b; } arr[N]` →
  `struct { float a[N], b[N]; }` so each field is a separate coalesced stream.

**Deeper fixes:**
- Use shared memory as a transposer: coalesced-load to shared, then
  arbitrary-access from shared.
- Vectorize: replace scalar loads with `float2` / `float4`.

**Exceptions:**
- Gather/scatter by random index (sparse matmul, embedding lookup) —
  fundamentally uncoalesced. Sort the indices for locality if possible.

---

## Pattern D — Sparse writes (low store efficiency)  ❌

**Signals (upstream):** `smsp__sass_average_data_bytes_per_sector_mem_global_op_st.ratio < 16`.

**MACA status:** not observable.

**What to do instead:** inspect the source for `if (lane_id < K) {
output[...] = ... }` patterns and confirm by re-timing after packing.

**First-line fix:** pack the write. Have the warp collectively produce `K`
values first (via shuffle or shared-memory reduction), then have exactly `K`
contiguous lanes perform `K` consecutive writes.

If `K >= 32` (or 64 on C500): all lanes can write; make sure the per-lane
index is contiguous.

If `K < 8`: consider batching multiple iterations' results into a vectorized
write (e.g. 4 iterations' output packed into a single `float4`).

**Deeper fixes:**
- Write into shared memory first, then do a coalesced global store at the end
  of the block.

**Exceptions:**
- Histogram / scatter (inherently sparse) — different optimization path, see
  Pattern G.

---

## Pattern E — Latency-bound (long-scoreboard-dominated)  ❌

**Signals (upstream):** `long_scoreboard` > 40% of samples, low DRAM
throughput, hotspots on global load lines.

**MACA status:** not observable. Use the proxy: a load-unrolling or
shared-memory staging experiment that moves the timing is the evidence.

**Why:** warps issue a load, then stall waiting for it to return before the
next dependent op. Usually combined with low occupancy or insufficient ILP.

**First-line fix:** increase in-flight memory requests:
- **Unroll the load loop** so 4-8 loads are issued before any value is used.
- **Add more independent warps** — raise occupancy (Pattern J).
- **Async bulk loads** — the MACA analogue of `cp.async`/TMA; check the SDK
  headers for the available intrinsic before assuming one exists.

**Deeper fixes:**
- Software pipelining: while tile N is being computed, pre-load tile N+1 into
  shared memory.
- Move reused data to shared memory so subsequent loads hit L1.

**Exceptions:**
- Pointer chasing / graph traversal — data dep chain is fundamental.

---

## Pattern F — Compute-bound but not on tensor cores  🔶

**Signals:**
- Roofline: `achieved_flops_pct_of_peak` high (Dim 4) — you must supply the
  op count.
- Workload is matmul-ish (GEMM, attention, conv).
- MACA cannot report tensor-core pipe activity, so "not on tensor cores" must
  come from inspecting the kernel source (no `wmma`/MMA intrinsics) or from
  the kernel name (`mcblas__Mck_tf32gemm_...` in the trace indicates the
  library path is using tensor cores).

**Why:** the kernel uses scalar FMA via the ALU pipe instead of tensor cores.

**First-line fix:** use `wmma`/`nvcuda::wmma` fragments or the MACA MMA
intrinsics. If hand-rolling is too much, use the MACA BLAS library
(`mcblas__...` kernels in the trace) which is already tuned.

**Deeper fixes:**
- Restructure data layout to meet MMA tile-shape constraints (e.g. `m16n8k16`
  for BF16).
- Use shared-memory staging.

**Exceptions:**
- Non-matrix workloads (reduction, sort, element-wise) — tensor cores don't
  help.
- Small matrices (M, N, K < 32) — tensor-core tiles are too coarse.

---

## Pattern G — Atomics contention  ❌

**Signals (upstream):** `long_scoreboard` on `ATOM`/`RED` SASS, high
`lts__t_sectors_op_atom`.

**MACA status:** not observable. Infer from source inspection (atomic ops in
a hot loop) and confirm with a hierarchical-reduction experiment.

**Why:** many threads atomically updating few locations → serialization.

**First-line fix:** hierarchical reduction.
- Within-warp: `__shfl_down_sync` (no atomic).
- Within-block: shared-memory reduction (no atomic).
- Between blocks: single atomic at the end per block.

**Deeper fixes:**
- Shared-memory histogram that flushes to global in one coalesced pass.
- Bucketing: thread writes to `output[tid % N_buckets]`, followed by a merge
  kernel.

**Exceptions:**
- Communication-library kernels — atomics are fundamental there.

---

## Pattern H — Shared-memory bank conflicts  ❌

**Signals (upstream):** `short_scoreboard` on shared-memory load lines.

**MACA status:** not observable. Reason from the access pattern in source.

**Why:** shared memory has 32 banks; same-bank accesses serialize. (C500
warp size is 64 — check the MACA programming guide for the bank count and
conflict behavior on this arch before assuming 32.)

**First-line fix:** padding. `__shared__ float tile[32][33]` instead of
`[32][32]` breaks regular bank alignment.

**Deeper fixes:**
- Swizzle: XOR-scramble indices so accesses spread across banks.
- Restructure data layout so warp lanes access different banks.

**Exceptions:**
- Broadcast reads (all lanes read same address) are conflict-free.
- Low shared-mem access volume — don't bother.

---

## Pattern I — Synchronization overhead  ❌

**Signals (upstream):** `barrier` stalls > 20% of samples, `BAR.SYNC` on the
hotspot line.

**MACA status:** not observable. Infer from `__syncthreads()` count in source
and confirm with a sync-reduction experiment.

**Why:** `__syncthreads()` waits for the slowest warp. Combined with any
per-warp work imbalance, this amplifies.

**First-line fix:**
- Replace block-level syncs with warp-level primitives (`__shfl_sync`,
  `__ballot_sync`, `__syncwarp`) where only warp-scoped synchronization is
  needed.
- Reduce total sync count — consolidate multiple synchronized phases.

**Deeper fixes:**
- Warp-specialized execution: producer warps and consumer warps with
  mbarrier instead of `__syncthreads`.

---

## Pattern J — Low achieved vs theoretical occupancy  🔶

**Signals:**
- Theoretical occupancy (derived) high, but the runtime-reported
  `mtreg_occupancy(%)` / `shared_memeory_occupancy(%)` low.
- On MACA you cannot measure *achieved* occupancy (no `sm__warps_active`), so
  this pattern is about the limit gap, not runtime behavior.

**Why:** theoretical occupancy is the max warps that *could* be resident. The
runtime limit tells you which resource binds first.

**First-line fix:** whichever limit binds — registers (Pattern K), shared
memory (reduce tile size), or block count (`__launch_bounds__` to raise the
block-per-SM limit).

---

## Pattern K — Register pressure / spill  🔶

**Signals:**
- `launch__registers_per_thread` high (>128 on C500 with 131072 regs/SM).
- Occupancy limited by `launch__occupancy_limit_registers`.
- MACA cannot detect actual spill (no `local_ld` counters), so high register
  count + low occupancy is the visible signal, not proven spill.

**Why:** too many live values forces the compiler to use more registers per
thread, reducing how many warps can be resident per SM.

**First-line fix:** `__launch_bounds__(maxThreadsPerBlock,
minBlocksPerMultiprocessor)` on the kernel. This tells the compiler to stay
within a register budget.

**Deeper fixes:**
- Reduce the number of live values: recompute values instead of caching,
  split the kernel into two.
- Move per-thread arrays to shared memory with explicit indexing.

**Exceptions:**
- Large fused kernels (FlashAttention) accept lower occupancy in exchange for
  larger savings upstream.

---

## Pattern L — FP64 used unintentionally  🔶

**Signals:** kernel should be FP32, but is slower than expected. MACA cannot
report FP64 pipe activity.

**Why:** C/C++ floating-point literals (`1.0`, `0.5`, `3.14`) default to
`double`. A `float x = a + 1.0 * b;` promotes `a + 1.0*b` to double.

**First-line fix:** add `f` suffix to all literals: `1.0f`, `0.5f`, `3.14f`.
Add `__expf` / `__logf` / `__sinf` variants for transcendentals.

**How to verify:** compile with `-save-temps` and inspect for double-precision
instructions, or simply apply the fix and re-time.

---

## Pattern M — Pipeline bubbles (no compute/memory overlap)  ✅ (run-level)

**Signals:**
- Timeline: gaps between compute phases and memory phases, visible as
  alternating idle/busy bands in the in-flight depth plot.
- `queue_ts` gaps: host-side work between launches.

**Why:** kernel loads a tile, computes on it, loads next tile —
single-buffered.

**First-line fix:** double-buffer. Use two shared-memory tiles; while
computing on tile A, load tile B. `__syncthreads` between phases.

**Deeper fixes:**
- Multi-stage pipeline (3-4 stages). Use async loads where available.

---

## Pattern N — Warp divergence  ❌

**Signals (upstream):** `smsp__thread_inst_executed_per_inst_executed.ratio`
far from the warp size.

**MACA status:** not observable. Inspect branches in source, especially ones
where the condition depends on thread index or data.

**Why:** lanes in a warp take different paths at a branch; hardware
serializes.

**First-line fix:**
- Rearrange so all lanes in a warp take the same branch. Sort / partition
  data if possible.
- Convert `if (cond) a else b` to branchless `mask * a + (1-mask) * b` —
  cheap if both sides are cheap.

**Exceptions:**
- Tree reductions in warps (last few steps have half / quarter / ... active).
  Use `__shfl_down_sync` to handle cleanly.
- Boundary handling (a few warps at tensor edge) — not worth fighting.

---

## Ranking template for the final report

When you hand back an optimization plan, rank by `(expected speedup) ×
(effort ratio)`. Without ncu's `Est. Speedup`, the estimate comes from wave
math, time share, and the experiments you ran — show the arithmetic.

```
Priority 1: <pattern> — <concrete fix>
  Evidence: <field values>
  MACA observability: ✅ / 🔶 / ❌
  Estimated impact: <backed by experiment E# or wave math>
  Effort: <low / medium / high>
  Why now: <reason this is the highest-leverage fix>

Priority 2: ...
```

A good rule of thumb: at most 3-5 priorities in the plan. More than that
dilutes the signal, and priorities > 5 usually contribute < 5% speedup each.
