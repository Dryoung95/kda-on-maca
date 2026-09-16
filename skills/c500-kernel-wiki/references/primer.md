# Topic Map — Symptom → Page

The entry point when you know what's wrong but not which page covers it. Each
row is a symptom a kernel author actually observes on C500, with the page that
explains it and the misconception it corrects.

## Silent wrong answers (check these first)

These fail without raising errors. A correctness gate passing on one seed does
not rule them out.

| Symptom | Cause | Page |
|---|---|---|
| Reduction returns half the expected sum | 5-round warp reduction assumes 32-lane warps; C500 has 64 | [warp64-implications](../wiki/hardware/warp64-implications.md) |
| Result matrix is a valid but wrong product | A/B operand swap in MMA — C500 calls are B-first in *some* mctlass traits, A-first in others | [mma-and-tensor-cores](../wiki/hardware/mma-and-tensor-cores.md) |
| Kernel skips a sync and gives intermittent results | `grid_group::is_valid()` is false on C500; guarded code silently skips | [cuda-compatibility-matrix](../wiki/migration/cuda-compatibility-matrix.md) |
| `__shfl_sync` affects only half the warp | 32-bit mask `0xffffffff` on a 64-lane warp | [warp64-implications](../wiki/hardware/warp64-implications.md) |
| Half the lanes of a ballot mask are zero | Same: 32-bit return on a 64-lane warp | same |

## Performance symptoms

| Symptom | Likely cause | Page |
|---|---|---|
| Matmul-shaped kernel 100× slower than torch | Scalar FMA instead of tensor cores; 0.17 vs 87 TFLOPS | [mma-and-tensor-cores](../wiki/hardware/mma-and-tensor-cores.md) |
| Tiled kernel slows 3.5× at stride 32 | Write-address collision (6.4× on writes, flat on reads), multiplied by the barrier — not a read bank conflict | [shared-memory-banks](../wiki/hardware/shared-memory-banks.md) |
| Kernel slower than its arithmetic suggests | Arithmetic intensity below the ~72 FLOPs/byte crossover | [c500-rooflines](../wiki/hardware/c500-rooflines.md) |
| Small grid underfills the GPU | 104 SMs; check waves/SM | [c500-architecture-facts](../wiki/hardware/c500-architecture-facts.md) |
| Block-size choice feels arbitrary | 512 measured best for elementwise; 32 costs 2.5× | [warp64-implications](../wiki/hardware/warp64-implications.md) |
| Many small kernels each cost a few µs | Launch overhead; fuse | [c500-rooflines](../wiki/hardware/c500-rooflines.md) |
| Barrier-heavy kernel slower than expected | `__syncthreads` ~0.03–0.16 µs each; batched barriers amortize | [shared-memory-banks](../wiki/hardware/shared-memory-banks.md) |

## Porting symptoms

| Symptom | Cause | Page |
|---|---|---|
| `fatal error: '__macro_mxcc.h' file not found` | cucc needs cu-bridge include path — environment, not source | [cuda-compatibility-matrix](../wiki/migration/cuda-compatibility-matrix.md) |
| `cp.async` / TMA / `wgmma` code won't compile or runs as no-op | These are dead code or disabled on C500 | [feature-mapping](../wiki/migration/feature-mapping.md) |
| Hopper strategy from a playbook does nothing | That playbook target does not exist here | [feature-mapping](../wiki/migration/feature-mapping.md) |
| OOM on a workload that fit elsewhere | 16.3 GB total; evaluation holds input + ref + new outputs | [c500-rooflines](../wiki/hardware/c500-rooflines.md) |
| `ldmatrix` fragment load behaves wrong | Disabled; use `__shfl_down_sync` reshuffle | [mma-and-tensor-cores](../wiki/hardware/mma-and-tensor-cores.md) |

## Diagnosis order

When a C500 kernel is slow and the cause is unknown, this order tends to find it
fastest:

1. **Compare against the ceilings.** 87 TFLOPS tf32 / 194 TFLOPS bf16 / 1.49 TB/s.
   If you are far below all of them, the problem is not the algorithm.
2. **Check the launch geometry** with the mctracer skill — small grids and
   wrong block sizes dominate everything else.
3. **Check for the silent failures** above. A kernel that gives wrong numbers
   is often a warp-reduction bug, not a slow kernel.
4. **Check arithmetic intensity** against the 72 FLOPs/byte crossover. Below it,
   the kernel is memory-bound and no amount of compute tuning helps.
5. **Only then** dig into access patterns and barriers.

## What this map does not cover

Kernel-specific algorithmic questions (which attention variant, which tiling)
are out of scope — those are design decisions, not platform facts. Use the
mctracer skill to gather evidence about your specific kernel, then use these
pages to interpret it.
