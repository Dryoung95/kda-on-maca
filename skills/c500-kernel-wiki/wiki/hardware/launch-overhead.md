---
title: Kernel Launch Overhead on C500
tags: [launch-overhead, short-kernels, fusion]
architecture: [c500]
type: hardware
provenance: measured
measured: 2026-09-15
probe: probes/probe_launch.cu
confidence: high
---

# Kernel Launch Overhead

Measured 2026-09-15. Empty kernel (`if (threadIdx.x == 0) out[0] = 1.0f;`), timed
over 500–1000 launches in a single cudaEvent window.

## The number

**~3.0–3.4 µs per launch** for grids up to one wave. This is floor cost — it is
paid by every kernel regardless of how little work it does.

| Grid (block 64) | µs/launch |
|---:|---:|
| 1 | 3.16 |
| 8 | 2.97 |
| 64 | 3.43 |
| 104 | 3.42 |
| 416 | 3.72 |
| 1664 | 5.17 |

Launch cost is roughly flat through one wave (104 blocks) and then climbs with
grid size — 5.17 µs at 1664 blocks. The interesting regime is the flat one: **a
kernel that does less than ~3 µs of work is mostly paying for being launched.**

## Why this dominates small-kernel decisions

Compare against measured kernel times on this board:

| Kernel | Duration | Launch share |
|---|---:|---:|
| saxpy, 1M elements | 16.8 µs | 19% |
| matmul 1024² | 60 µs | 5% |
| A small fused op on a short vector | ~1 µs | ~75% |

Small-batch or short-sequence work — attention decode at batch 1, a reduction
over a small tensor, a fused activation — sits in the regime where launch
overhead is the majority of runtime. This is exactly the case that appeared in
the CUDA-Agent migration: the axpby benchmark at 1×128 had ~9.4 µs of launch
overhead swamping the kernel, leaving the reward signal with no discrimination.

The fix is always the same shape: **fuse more work into one launch**.

- Merge the elementwise op into the producing kernel's epilogue.
- Batch N independent launches into one grid with an extra index dimension.
- Persistent kernel + work queue when the per-item work is truly tiny
  (though note [grid_group sync is unavailable](../migration/cuda-compatibility-matrix.md),
  so the work queue needs an atomic counter, not a grid barrier).

## What does not help

- **CUDA graphs** reduce *host-side* dispatch cost and are supported, but the
  ~3 µs measured here is the device-side launch floor, which graphs do not
  remove. Graphs help when the host is the bottleneck; check where the time
  actually is before assuming.
- **Stream parallelism** does not hide a per-launch floor; it interleaves it.

## In the trace

In an mcTracer trace, `submit_ts - queue_ts` captures the host-side enqueue cost
and `dur` the device execution. The gap between consecutive launches' `ts`
values is where launch overhead lives. If that gap is comparable to `dur`,
fusion is the highest-value change available — higher than any instruction-level
optimization of the kernel body.

See [c500-rooflines](c500-rooflines.md) for what a well-tuned kernel actually
achieves, and use the mctracer-report-skill to measure the gap on your kernel
rather than assuming it.
