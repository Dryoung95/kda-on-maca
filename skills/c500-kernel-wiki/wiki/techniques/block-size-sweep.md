---
title: Block-Size Sweep on C500
tags: [block-size, occupancy, launch-geometry, elementwise]
architecture: [c500]
type: technique
provenance: measured
measured: 2026-09-15
probe: probes/probe_blocksize.cu
confidence: high
---

# Block-Size Sweep

The cheapest high-yield experiment on this board. On NVIDIA-flavored CUDA the
default answer "256, multiple of 32" is usually fine; on C500 it is measurably
wrong, because the warp is 64 and the occupancy math does not transfer.

## The measurement

1M-element elementwise kernel, 50 reps in one cudaEvent window, block size
swept in powers of two (grid adjusted so total threads are constant):

| Block | Grid | µs/call | vs best |
|---:|---:|---:|---:|
| 32 | 32768 | 41.76 | 2.45× |
| 64 | 16384 | 23.07 | 1.36× |
| 128 | 8192 | 19.97 | 1.17× |
| 256 | 4096 | 18.93 | 1.11× |
| **512** | 2048 | **17.02** | **1.00×** |
| 1024 | 1024 | 17.61 | 1.03× |

**512 wins** — 8 warps per block. The 2.45× penalty at 32 threads is large
enough that the sweep pays for itself before any other optimization.

## Why 512, and why not to generalize

A 512-thread block at warp 64 is 8 warps. The NVIDIA-trained instinct is 256
threads = 8 *warps*; on C500 that same instinct yields 4 warps, half the
concurrency per block. But the optimum is a property of *this workload class*
(memory-bound elementwise), not of the board:

- This probe is memory-bound with no shared memory and no barriers. A kernel
  with 64 KB of shared memory per block or a large register footprint hits a
  different occupancy limit and its optimum moves.
- 1024-thread blocks are within 3% of the winner here but cost all 2048
  threads/SM of occupancy in one block — leave the room if the kernel might
  need it.
- Reductions and scan have their own optima because the tree depth changes with
  block size. Sweep, do not reuse this table.

## How to run it on your kernel

```bash
source /data/kda-maca/env.sh
cucc probe_blocksize.cu -O2 -std=c++17 $MACA_CUCC_FLAGS -o probe_blocksize
./probe_blocksize
```

Adapt the probe's kernel body to your own work, keep the sweep and the timing
harness, and compare ratios rather than absolute times. The ratio table
transfers between boards with different clocks; the microsecond values do not.

## What a bad result pattern means

| Observation | Read it as |
|---|---|
| Flat from 128 upward | The kernel is launch- or memory-bound; block size is not the lever |
| Monotonic improvement to the max | Occupancy-starved; check shared memory and register limits |
| Best at 32 | Likely warp-divergence-bound; a 32-thread block is half a C500 warp |
| Non-monotonic with a spike | Interaction with barriers or shared-memory capacity — profile with [mctracer](../../../mctracer-report-skill/SKILL.md) |

## Related

- [warp64-implications](../hardware/warp64-implications.md) — why the 32-centric
  defaults are wrong at warp 64
- [launch-overhead](../hardware/launch-overhead.md) — if the sweep is flat, the
  binding constraint may be the launch, not the geometry
