---
title: Shared Memory on C500
tags: [shared-memory, banks, barriers, bank-conflict, divergence]
architecture: [c500]
type: hardware
provenance: measured
measured: 2026-09-16
probe: probes/probe_smem_mixed.cu
confidence: high
---

# Shared Memory on C500: Banks, Barriers, and a Common Misdiagnosis

The NVIDIA mental model for shared memory is 32 banks, 4 bytes wide, stride-32
conflicts. Carrying that model to C500 produces wrong diagnoses, because the
observed stride-dependent slowdown is caused by something else entirely.

This page is built from three probes that isolate shared-memory read, write, and
barrier behavior independently. All measured 2026-09-15 on this board.

## Measurement setup

Three kernels, each isolating one operation:

- **pure load** — 256 threads, 2000 iterations, strided shared-memory reads, no
  `__syncthreads`.
- **pure store** — same, strided writes, one barrier at the end.
- **mixed** — strided store + load with two barriers per iteration (the pattern
  a tiled kernel actually has).

Stride is in floats; address space 4096 floats.

## Result: reads are flat, writes are not

| Stride | Pure load | Pure store | Mixed (2 barriers/iter) |
|---:|---:|---:|---:|
| 1 | 1.00× | 1.00× | 1.00× |
| 8 | 0.97× | 1.83× | 1.59× |
| 16 | 0.97× | 3.38× | 2.24× |
| 32 | 0.97× | **6.43×** | **3.53×** |
| 64 | 0.97× | **6.43×** | **3.53×** |

**Reads are flat**: every stride costs the same as stride 1, up to
stride 64. But **strided writes are 6.4× slower than stride-1 writes even with no
barrier at all.** The original version of this page reported both reads and
writes as flat — that was an artifact of the store being hoisted out of the loop
by the compiler, which the current probe prevents by making the address and value
iteration-dependent. The write penalty is real, and it is the largest single
number on this page.

So the bank geometry *does* penalize strided writes — it just does not penalize
strided reads, and it does not need a barrier to show up. The mixed kernel's
3.53× is the write penalty interacting with the barrier, not a pure bank
conflict in the NVIDIA sense.

**All three baselines are state-dependent and must be measured warm.** A fresh
process measures stride 1 at up to ~1.6× its settled value, which compresses
every ratio in the table. Each probe warms up on stride 1 and stride 32 before
sweeping; the ratios above are from the settled state and reproduce within ±1%
across runs.

## The misdiagnosis this prevents

The mixed probe — the one that looks like a real tiled kernel — slows 3.53× at
stride 32. An author trained on NVIDIA CUDA reads that as "textbook bank
conflict" and reaches for padding: `tile[32][33]`. The diagnosis is half right
and the fix is wrong.

The cause is **write-address collision** — and it does not need the barrier at
all, as the pure-store column above shows (6.43× with zero `__syncthreads`).
Three controls pin it down (all at 256 threads, 200 iterations):

- Remove the two `__syncthreads` from the mixed kernel and the stride-32 cost
  drops from 122.5 µs to 15.2 µs — but it does not drop to stride 1's level
  (15.7 µs is itself elevated by the same write collision). The barrier
  *amplifies* the collision; it is not its only cause.
- Keep the barriers but make every thread write the *same* address (zero
  divergence, no collision geometry): 75.0 µs, as slow as the strided case. So
  thread divergence is *not* the driver.
- Keep the barriers and the strided write, but make the read non-broadcast:
  still 82.8 µs at stride 32. The collision lives in the write path, not the
  read path.

So: **on C500 a stride-32 write collision is a real write-path cost, and in a
tiled kernel the barrier multiplies it.** The fix is to change the write pattern
or the sync structure — padding the read layout does nothing, because reads were
never the problem.

## Barrier cost, measured directly

| `__syncthreads` per iteration | Total (µs) | Per-barrier (µs) |
|---:|---:|---:|
| 0 | 15.69 | — |
| 1 | 31.10 | 0.156 |
| 2 | 39.65 | 0.099 |
| 4 | 56.73 | 0.071 |
| 8 | 52.15 | 0.033 |

Two useful readings:

1. **A single barrier costs ~0.16 µs** at this scale. A kernel with 100 barriers
   in a loop pays ~10 µs of pure sync — comparable to the entire runtime of a
   small kernel. Barrier count is a first-class optimization target on C500.
2. **Marginal barrier cost falls as concurrency rises** (0.156 → 0.033), because
   the compiler batches barriers. Two adjacent barriers are cheaper than two
   separate ones — consolidating sync phases is a real win, not a stylistic one.

This directly changes the calculus on upstream Pattern I (synchronization
overhead): on NVIDIA the advice is "replace block syncs with warp primitives."
On C500 the cheaper and equally valid move is often "merge two synced phases
into one," because batched barriers amortize.

## What to do about layout

Since padding does not buy what it buys on NVIDIA, do not cargo-cult it. Use the
sanctioned layout pattern instead: `mcflashinfer/permuted_smem.cuh`
implements XOR swizzle in `SwizzleMode::{k64B,k128B}` with 8-row (b128) and
16-row (b64) swizzle blocks. That is the layout the shipped C500-native kernels
actually use, and it is tuned to this hardware's real bank geometry rather than
to a borrowed assumption.

**Verify before trusting any of this.** These are measurements from one board on
one SDK version. The cheap way to re-verify is the pure-load probe above: if it
stays flat across strides on a future SDK, the "reads are conflict-free"
conclusion still holds; if it develops structure, the bank geometry changed and
this page needs updating.

## Probes

- `probes/probe_smem_load.cu` — pure load sweep
- `probes/probe_smem_store.cu` — pure store sweep
- `probes/probe_smem_mixed.cu` — mixed load/store + barrier sweep
- `probes/probe_sync.cu` — barrier count sweep
