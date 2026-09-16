---
title: Warp Size 64 — What Actually Changes
tags: [warp64, warp-size, shuffle, reduction, block-size]
architecture: [c500]
type: hardware
provenance: measured
measured: 2026-09-15
probe: probes/probe_blocksize.cu
confidence: high
---

# Warp Size 64: What Actually Changes

The single highest-impact divergence from NVIDIA-flavored CUDA on this board.
`warpSize = 64` is a compile-time constant, not a query, so there is no runtime
path that discovers it — code either accounts for it or is silently wrong.

## The arithmetic that changes

| Quantity | NVIDIA (32) | C500 (64) |
|---|---:|---:|
| Full-warp coalesced group | 32 lanes / 128 B | **64 lanes / 256 B** |
| `__shfl` / `__ballot` mask width | 32 bits | 64 bits |
| Reduction tree depth in a warp | 5 steps | **6 steps** |
| Active-lane divergence cost | per 32 lanes | per 64 lanes |
| Warp-size assumption in `minThreadsPerBlock` heuristics | 32 | 64 |

Every warp-level reduction written as 5 shuffle steps stops one step early on
C500 and returns a half-reduced result — correct-by-luck only for symmetric
inputs where the missing step is an identity.

## Correct reduction pattern

```cpp
// WRONG on C500: assumes 5 rounds for a 32-lane warp
for (int offset = 16; offset > 0; offset >>= 1)
    v += __shfl_down_sync(0xffffffffffffffffULL, v, offset);

// RIGHT: derive from warpSize, works on both
for (int offset = warpSize / 2; offset > 0; offset >>= 1)
    v += __shfl_down_sync(0xffffffffffffffffULL, v, offset);
```

Use `warpSize` (it is a compile-time device constant, the compiler folds it) or
`__MACA_ARCH__`-guarded constants. Never write `16` or `32` as a reduction depth.

Mask literals are the other silent-failure site: `0xffffffff` is a 32-bit mask
that leaves the top 32 lanes out of a `__shfl_sync`. On C500 the full mask is
`0xffffffffffffffffULL`.

## Block-size heuristics

Measured on this board (1M-element elementwise kernel, 50 reps, cudaEvent):

| Block size | Time (µs) | vs best |
|---|---:|---:|
| 32 | 42.13 | 2.51× |
| 64 | 22.57 | 1.34× |
| 128 | 20.01 | 1.19× |
| 256 | 18.14 | 1.08× |
| **512** | **16.81** | **1.00×** |
| 1024 | 17.10 | 1.02× |

C500 prefers 512-thread blocks for memory-bound elementwise work — 8 warps per
block, which is a different optimum than the 256 that CUDA-trained authors reach
for by default. The 2.5× penalty at 32 threads is large enough that a
block-size sweep is worth running before any other optimization.

**Do not** copy the common CUDA heuristic "block size should be a multiple of
32 and usually 256". Derive it per kernel type with a sweep
(see [block-size sweep](../techniques/block-size-sweep.md)).

## Where 64 helps

Warp 64 is not a pure disadvantage: fewer warps per block means less scheduler
overhead per block, and each warp covers twice the data per program counter, so
loop-control overhead in grid-stride patterns halves. Kernels that are
launch-overhead-bound or loop-bound benefit; kernels that are warp-divergence-
bound suffer, because one divergent branch now wastes up to 64 lanes.

## What to grep for

Auditing a CUDA-flavored kernel for C500:

- `16`, `32` as reduction offsets or shuffle widths → derive from `warpSize`.
- `0xffffffff` as a shfl/ballot mask → extend to 64-bit.
- `31` as a lane mask (`laneId & 31`) → `& (warpSize-1)`.
- `5` as a reduction round count → 6, or derived.
- `>= 32` / `< 32` divergence guards sized to NVIDIA warps.
- Shared-memory tile widths of 32 or 33 (classic bank padding) — the padding
  logic needs revisiting for 64-wide groups; see
  [shared-memory-banks](../hardware/shared-memory-banks.md).
