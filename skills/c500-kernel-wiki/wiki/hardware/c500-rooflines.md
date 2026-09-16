---
title: C500 Performance Ceilings
tags: [roofline, bandwidth, flops, memory-bound]
architecture: [c500]
type: hardware
provenance: measured
measured: 2026-09-15
probe: torch.benchmark (in-page script)
confidence: high
---

# C500 Performance Ceilings (measured)

The numbers to divide by. Every "my kernel is slow" question on this board starts
here: what fraction of achievable throughput is it actually getting?

Measured 2026-09-15 on this host, torch 2.8.0+metax, MACA SDK 3.3.0.15. These are
*library-achieved* ceilings — the throughput a well-tuned kernel reaches — not
theoretical peaks. A hand-rolled kernel should be compared against these, not
against vendor marketing numbers.

## Compute ceilings

| Precision | Shape | Time | Achieved | Notes |
|---|---|---|---|---|
| fp32 (tf32 path) | 1024² | 0.06 ms | 37.2 TFLOPS | launch-bound |
| fp32 (tf32 path) | 2048² | 0.25 ms | 69.2 TFLOPS | |
| fp32 (tf32 path) | 4096² | 1.57 ms | 87.7 TFLOPS | |
| fp32 (tf32 path) | 8192² | 10.23 ms | **107.5 TFLOPS** | largest, most stable |
| bf16 | 1024² | 0.03 ms | 67.5 TFLOPS | |
| bf16 | 2048² | 0.12 ms | 138.1 TFLOPS | |
| bf16 | 4096² | 0.71 ms | **194.3 TFLOPS** | peak observed |

Working ceilings to remember: **~107 TFLOPS fp32/TF32, ~194 TFLOPS bf16**. The
bf16:fp32 ratio is ~1.8×, not the 2× that NVIDIA TF32-vs-BF16 math predicts, so
bf16 conversion buys slightly less than expected.

**Size matters more than it should.** Throughput doubles from 1024² to 8192² on
the same precision — small GEMMs are launch- and tiling-bound, not compute-bound.
Quoting a TFLOPS number without the shape it was measured at is meaningless.

## Memory bandwidth ceiling

| Workload | Bytes moved | Effective bandwidth |
|---|---|---|
| elementwise copy | 1 GB | 1.49 TB/s |
| elementwise copy | 2 GB | 1.49 TB/s |
| elementwise copy | 4 GB | 1.49 TB/s |

**1.49 TB/s sustained**, counted as *bytes read* per second (a copy moves the
same number of bytes again in writes, so a read+write accounting gives roughly
half this figure — ~1.4 TB/s on this board). Stable across sizes, so it is a
genuine bandwidth limit rather than a launch-overhead artifact. Quoting a
bandwidth number without saying which traffic you counted is meaningless.

Note this is well below typical HBM figures for competing data-center GPUs, which
makes C500 relatively more memory-bound per FLOP. A kernel at 100 TFLOPS on
matmul-shaped work and 1.49 TB/s on streaming work means the arithmetic intensity
crossover is at roughly **72 FLOPs/byte** — compute-bound only above that.

```
crossover = 107e12 / 1.49e12 ≈ 72 FLOPs/byte
```

Compare A100's ~10 FLOPs/byte crossover: on C500, far more kernels sit on the
memory side of the roofline. Prioritize data movement over instruction count in
almost every optimization decision.

## How to use these numbers

The roofline question for a C500 kernel is not "is it memory-bound" but "how far
from the measured ceiling is it." Compute your kernel's arithmetic intensity,
then:

```
achievable_flops = min(kernel_flops_per_byte * 1.49e12, 107e12)
efficiency = achieved_flops / achievable_flops
```

Efficiency under ~40% means the binding constraint is something else — launch
geometry, barriers, or divergence. Start with the
[block-size sweep](../techniques/block-size-sweep.md) and the
[mctracer timeline](../../../mctracer-report-skill/SKILL.md) before reaching for
algorithmic changes.

## The 16.3 GB constraint

Total memory is 16.3 GB. Input, output, and workspace must fit three-deep during
evaluation (reference output and candidate output both live on-device). Practical
budget per tensor: **under 4 GB**, under 5 GB at the absolute outside.

Workloads sized for 48 GB boards (large KernelBench problems, big-batch attention)
OOM here. Classify that as a capacity limit, not a backend defect — resizing the
workload is the fix, and a knowledge page cannot change the silicon.

## Measuring your own kernel against these

```python
import torch, time

def throughput(fn, flops, iters=20):
    for _ in range(3): fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters): fn()
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / iters
    return dt, flops / dt
```

Always warm up, always synchronize outside the window, always report the shape.
For per-launch detail instead of wall time, trace with mcTracer and read the
`dur` field — see the [mctracer-report-skill](../../../mctracer-report-skill/SKILL.md).
