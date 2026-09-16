---
title: NVIDIA Feature → C500 Substitute
tags: [migration, feature-mapping, hopper, blackwell]
architecture: [c500]
type: migration
provenance: header
confidence: high
---

# NVIDIA Feature → C500 Substitute

An optimization playbook written for NVIDIA GPUs names features that do not exist
here. This page is the translation table: for each Hopper/Blackwell-era feature
an upstream strategy might invoke, what to use instead on C500 — or that nothing
exists.

Established from the cute/mctlass headers shipped in MACA SDK 3.3.0.15. The
"status" column reflects what the *toolchain* enables, not what the hardware
might theoretically support.

## The load-bearing fact

C500's effective capability is **SM75/SM80-era, plus MACA-native builtins.** The
cute arch tree ships files for `sm70`–`sm90` only, and every SM90 feature is
present as dead code with its enable macro commented out. There is no
xcore1000-named arch file; C500 code is grafted onto the `Sm80` tag and gated on
`__MACA_ARCH__ == 1000`.

## Mapping table

| NVIDIA feature | C500 status | Substitute |
|---|---|---|
| `cp.async` (SM80) | ❌ disabled — `copy_sm80.hpp: #if defined(__MACA_ARCH__) && 0` | `__builtin_mxc_ldg_b{32,64,128}_bsm[_predicator]` |
| `ldmatrix` (SM75) | ❌ `CUTE_ARCH_LDSM_SM75_ACTIVATED` off | `__shfl_down_sync` fragment reshuffle (as mctlass does) |
| `stmatrix` | ❌ same family | same reshuffle, reversed |
| TMA (SM90) | ❌ `CUTE_ARCH_TMA_SM90_ENABLED` commented | no bulk-copy descriptor path; stage explicitly in shared memory |
| `wgmma` (SM90) | ❌ `CUTE_ARCH_MMA_SM90A_ENABLED` commented | `__builtin_mxc_mma_16x16x*` (warp-level, not warpgroup) |
| Thread block clusters (SM90) | ❌ `CUTE_ARCH_CLUSTER_SM90_ENABLED` commented | none — no cross-block cooperation beyond atomics |
| `mbarrier` | ❌ only referenced inside disabled TMA files | `__syncthreads` (batched — see [shared-memory-banks](../hardware/shared-memory-banks.md)) |
| `tcgen05` (Blackwell) | ❌ absent entirely — zero hits for `tcgen` | none |
| TMEM accumulators | ❌ absent | register fragments |
| `mma.sync m16n8kK` | ❌ wrong shape | `__builtin_mxc_mma_16x16x*` — 16x16xK, **B-first** |
| 2-SM cooperative (Blackwell) | ❌ absent | single-SM tiling |
| PDL / GDC | ❌ absent | — |

## Async arrival tracking

The MACA arrival builtins replace Hopper's mbarrier-based producer/consumer
synchronization. From `maca_kernel_utils.hpp`:

```
arrive_gvmcnt(count)  →  __builtin_mxc_arrive(64 + count)
arrive_bsmcnt(count)  →  __builtin_mxc_arrive(4096 + 128 * count)
```

`bsm` here is the MACA term for the local store backing the async load — it is
the closest thing to shared-memory staging with completion tracking. If you are
porting a double-buffered pipeline, this is the mechanism, and the
`gvmcnt`/`bsmcnt` counts are the producer/consumer signals.

## What this means for playbook strategies

Any advice that names a Hopper or Blackwell feature is inapplicable verbatim:

- "use TMA for bulk loads" → stage loads into shared memory explicitly, use the
  `ldg_b*_bsm` builtins if async is needed.
- "use wgmma with TMEM accumulators" → use warp-level `__builtin_mxc_mma_16x16x*`
  with register fragments.
- "use clusters for 2-SM cooperation" → restructure to single-SM or accept
  atomic-based coordination.
- "warp-specialize with mbarrier" → warp-specialize with `__syncthreads`, which
  on C500 is cheap enough (~0.03 µs marginal when batched) to be a reasonable
  substitute.
- "persistent kernel with grid sync" → **not available**; grid_group is invalid
  (see [cuda-compatibility-matrix](cuda-compatibility-matrix.md)).

## What you keep

SM80-era strategies port well and are the right mental model:

- Shared-memory tiling with double buffering (using the `bsm` builtins for the
  load side).
- Warp-level MMA with fragment reshuffles.
- `__launch_bounds__` for register budgeting — verified working.
- Swizzled shared-memory layouts (use mcflashinfer's `permuted_smem.cuh`
  `SwizzleMode::{k64B,k128B}` rather than NVIDIA's swizzle helpers).
- Library-first for GEMM-shaped work: mcBLAS and the prebuilt mctlassEx kernels
  are C500-native.

## Verifying a feature is really disabled

The macros are the source of truth, and they are worth grepping directly rather
than trusting this page, because they can flip between SDK releases:

```bash
grep -rn "CUTE_ARCH_TMA_SM90_ENABLED\|CUTE_ARCH_MMA_SM90A_ENABLED" \
    /opt/maca-3.3.0/include/cute/arch/
```

An enable macro that is uncommented means this page is out of date for that
feature — update it, and re-run the affected probes.
