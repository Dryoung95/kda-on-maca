---
title: Register Pressure and Occupancy on C500
tags: [registers, occupancy, register-spill, launch-bounds]
architecture: [c500]
type: hardware
provenance: measured
measured: 2026-09-15
probe: probes/probe_regs.cu
confidence: medium
---

# Register Pressure and Occupancy

How register count actually affects C500 kernels — and the measurement that
corrects the usual assumption. Marked `confidence: medium` because the most
important finding here is negative: the toolchain's register observability is
poor, and several intended experiments did not behave as expected.

## The intended experiment, and what actually happened

Design: a kernel parameterized on unroll factor `N`, holding `N` live values,
sweeping `N` from 4 to 128 and watching register count, reported occupancy, and
throughput. Four variants were tried (array with unroll, `volatile` array, named
accumulators, `-maxrregcount` clamping).

Measured throughput, block=256, 1M elements:

| Unroll / live values | µs |
|---:|---:|
| 4 | 19.4 |
| 8 | 19.1 |
| 16 | 18.5 |
| 32 | 20.1 |
| 64 | 33.2 |
| 96 | 47.4 |
| 128 | 60.9 |

There is a real, steep cost at high live-value counts — throughput degrades 3×
from unroll 16 to 128. That part is reliable and matches the mctlass observation
that the 128x128x256 i8 GEMM budgets ~160 registers for operands+accumulators.

## The correction: this is not an occupancy effect

Every variant, at every live-value count, reported **`registers_per_thread = 4`
and `mtreg_occupancy = 0%`** in the mcTracer trace. Including variants where
throughput differed by 3×.

That is not a measurement error in one probe — it is a consistent property of
what mcTracer reports for kernels built through cucc in this configuration:

- The reported register count does not track the kernel's live-value count.
- `-maxrregcount=N` is **not honored** by cucc in this setup — the count stays at
  its chosen value regardless of the flag.
- Occupancy fields (`mtreg_occupancy(%)`) report 0 for these kernels.

So the 3× throughput degradation is **not** occupancy loss in the sense NVIDIA
tooling would show. It is instruction-count and scheduling pressure: more live
values means more instructions to issue and a longer dependency chain, which the
mcTracer fields do not surface.

**Practical consequence:** on C500 you cannot diagnose register pressure the way
ncu's `launch__registers_per_thread` + `occupancy_limit_registers` pair lets you.
Do not attempt to read register pressure off the trace for arbitrary kernels —
the field is only meaningful for kernels the runtime instruments that way
(mctlassEx library kernels do report realistic values, e.g. 224 regs for the
mcBLAS GEMM).

## What to do instead

Since observation is unreliable, work from effect rather than cause:

1. **Measure throughput, not register count.** Sweep the unroll factor and plot
   duration. The cliff position (32→64 above) is the actionable number; the
   register count that produces it is not observable.
2. **`__launch_bounds__` is accepted and honored** (verified compiling), so use
   it to state the register budget even though you cannot verify the result by
   reading it back.
3. **Budget against mctlass's real figures.** The shipped 128x128x256 i8 GEMM
   uses ~160 registers for operands + accumulators
   (`maca_mma_multistage_i8_tn_128x128x256.h`), with 4-element fragment arrays
   per stage. If a hand-rolled kernel uses substantially more than that for
   comparable work, it is over-budget — compare against the reference rather
   than against an unobservable counter.

## Related limits (derived, from device constants)

With 131,072 registers per SM and 2048 max threads per SM:

| Threads/block | Blocks/SM by registers (if regs were as reported) |
|---|---|
| 64 | 32 (warp limit binds first at 2048/64) |
| 128 | 16 |
| 256 | 8 |
| 512 | 4 |

These are arithmetic from device properties, valid as ceilings. Because the
reported register count is unreliable, treat this table as a bound on what is
*possible*, not a claim about what any specific kernel achieves.

## Probes

- `probes/probe_regs.cu` — unroll sweep, throughput
- `probes/probe_regs2.cu`, `probes/probe_regs3.cu`, `probes/probe_regs4.cu`,
  `probes/probe_regs5.cu` — the register-count observability experiments
  (each documents a way the reported count failed to track reality)

The negative results are the point of this page: a porting agent that tries to
apply upstream Pattern K (register spill detection via `local_ld` counts) will
find no signal on this platform, and one that trusts `registers_per_thread` from
the trace will draw wrong conclusions.
