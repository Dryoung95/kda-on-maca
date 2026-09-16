---
name: c500-kernel-wiki
description: Use when optimizing or porting a CUDA kernel to the MetaX C500 (MACA, xcore1000) — warp size 64, tensor-core shapes, shared-memory behavior, CUDA compatibility pitfalls, and measured performance ceilings. Also use for questions like "why is my C500 kernel slow", "does X work on C500", or Chinese variants ("为什么慢", "C500 上能不能用 X").
---

# C500 Kernel Wiki

Domain knowledge for GPU kernel work on the MetaX C500 (MACA software stack,
xcore1000). This is the C500 replacement for the upstream KDA `KernelWiki`,
which is Blackwell/Hopper-only and has no content applicable to this board.

**When to use:** any question about writing, porting, or optimizing a kernel on
C500. Triggers include "port this kernel to C500", "why is my kernel slow on
C500", "does `cooperative_groups` work on MACA", "what block size should I use",
"为什么这个 kernel 在 C500 上慢".

**When NOT to use:** NVIDIA-specific questions (B200/H100 tensor cores,
Hopper wgmma, Blackwell tcgen05) — use the upstream KernelWiki instead. This wiki
has nothing for those targets.

## How to use

Read pages directly; this wiki is small enough to browse rather than query.

**Start from the symptom:**

| If you are wondering… | Read |
|---|---|
| What's different from CUDA at all | [cuda-compatibility-matrix](wiki/migration/cuda-compatibility-matrix.md) |
| Why a stride-32 shared-memory pattern is slow | [shared-memory-banks](wiki/hardware/shared-memory-banks.md) |
| Whether warp intrinsics need changing | [warp64-implications](wiki/hardware/warp64-implications.md) |
| If tensor cores can help | [mma-and-tensor-cores](wiki/hardware/mma-and-tensor-cores.md) |
| What throughput to expect | [c500-rooflines](wiki/hardware/c500-rooflines.md) |
| Which NVIDIA feature maps to what | [feature-mapping](wiki/migration/feature-mapping.md) |
| Absolute hardware facts | [c500-architecture-facts](wiki/hardware/c500-architecture-facts.md) |

**Start from the hardware:**

- [c500-architecture-facts](wiki/hardware/c500-architecture-facts.md) — 104 SM,
  warp 64, `__MACA_ARCH__ == 1000`, what exists and what is dead code
- [c500-rooflines](wiki/hardware/c500-rooflines.md) — 87 TFLOPS tf32, 194 TFLOPS
  bf16, 1.49 TB/s, and the arithmetic-intensity crossover they imply

**Start from the misconception** (the highest-value pages, because they correct
assumptions that fail silently):

- Warp size is 64, and a 5-round warp reduction is silently wrong.
- Strided shared-memory *writes* cost 6.4× on C500 while *reads* are flat, and
  the barrier multiplies it. Padding the read layout is the wrong fix — it
  addresses the side that was never slow.
- `cooperative_groups` grid sync is silently invalid.
- MMA is 16x16xK, and the operand order is B-first in *some* mctlass traits and
  A-first in others — copy the whole trait, not the builtin call line.

**Techniques:**

- [block-size-sweep](wiki/techniques/block-size-sweep.md) — the cheapest
  high-yield experiment on the board; 512 wins for elementwise, 32 costs 2.5×
- [mctlass-kernel-patterns](wiki/techniques/mctlass-kernel-patterns.md) — what
  the shipped C500-native GEMM actually does, for hand-rolling against

## Why this wiki is built from measurement

The SDK ships no kernel-optimization documentation. The samples tree contains 72
programs and zero warp-64, bank-conflict, occupancy, or SM-utilization content —
it is ported CUDA boilerplate that validates API parity and nothing more. Online
documentation for this board is not reliably reachable.

So the facts here come from microbenchmarks run on this exact board
(`probes/`), with the provenance recorded on every page. That makes the wiki
verifiable — any number can be reproduced by building and running its probe —
and it makes the limitations honest: a `measured` fact is a fact about one SDK
version, not about the architecture forever.

Every page says how it knows. If a page's claim would change with an SDK upgrade,
its frontmatter records the date and probe so you can tell which facts are at
risk. See [references/schema.md](references/schema.md).

## Conventions

- All numbers are measured on this host unless marked `header` (read from a
  toolchain file) or `derived`.
- Reproduce before quoting, especially after an SDK upgrade. Probes build with:
  `cucc probe_X.cu -O2 -std=c++17 $MACA_CUCC_FLAGS -o probe_X`.
- Silent failures get extra emphasis: on this platform the dangerous bugs are
  the ones that compile, run, and return wrong numbers.

## Related skills

- [mctracer-report-skill](../mctracer-report-skill/SKILL.md) — profiling and
  per-launch evidence on C500. This wiki tells you what to expect; that skill
  tells you what your kernel actually did.
