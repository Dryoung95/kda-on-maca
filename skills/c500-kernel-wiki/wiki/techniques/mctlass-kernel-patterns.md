---
title: mctlass GEMM Authoring Patterns for C500
tags: [mctlass, gemm, pipeline, double-buffering, epilogue, mma]
architecture: [c500]
type: technique
provenance: header
confidence: high
---

# mctlass GEMM Authoring Patterns

What the shipped C500-native GEMM actually does. mctlass is the MACA port of
CUTLASS, and its kernels are the closest thing to official "how to write a fast
C500 kernel" examples — they are prebuilt into
`lib/mctlassEx_xcore1000.mcfb` and run at measured library throughput.

This page is for anyone hand-rolling a tiled GEMM-ish kernel and wanting to match
the shipped implementation's structure rather than invent one.

## Negative findings first (these correct plausible assumptions)

| Assumption | Reality |
|---|---|
| There is a `mctlass/pipeline/` policy class for double-buffering | **No such directory exists.** Double-buffering is hand-scheduled inline in each kernel. |
| `cp.async` exists but is disabled | There is no NVIDIA `cp.async`; the MACA equivalent is `__builtin_mxc_ldg_b128_bsm_predicator` plus `maca_cp_async_zfill<N>` / `maca_cp_async_wait<N>` (`mctlass/arch/maca_memory.h:272-356`). |
| `cp_async_fenc()` is a real fence | **It is a no-op comment**: `#define cp_async_fenc() asm(";--------------");` (`maca_kernel_utils.hpp:27`). Ordering comes from the barrier counters. |

The pipeline abstraction is deliberately thin: mctlass overlaps load and compute
by interleaving instructions by hand in one unrolled body, not by invoking a
scheduling policy.

## Tile shapes and stages

The shipped configs (`mctlass/gemm/threadblock/maca_mma_multistage/`):

| Config | Tile (M×N×K) | kStage | smem |
|---|---|---|---|
| i8 tn | 128×128×256 | **4** | 64 KB (A,B one tile each; K-tile split into 4 staggered sub-chunks) |
| i8 tn | 32×32×256 | 2 | `(kSizeA + kSizeB) * kStages` (true double buffer) |
| i8 tn | 64×128×256 | 2 | `kSizeA + kSizeB/2` — asymmetric: A single, B halved (32 KB) |
| bf16/fp16 group | 128×128×128 | 4 | allocated outside the class |

The actionable pattern: **the highest-throughput config buys a larger K-tile
(256) with single-buffered operand smem plus 4-way sub-chunk staggering, rather
than a deeper buffer.** Smaller M-tiles spend smem on a genuine 2-stage buffer.
Stage depth is bounded by shared memory, and the big config spends its smem
budget on K-tile width instead of buffer depth.

## The pipeline: hand-scheduled barrier counting

Two counters drive everything (`maca_kernel_utils.hpp:6-7`):

```cpp
#define arrive_gvmcnt(count)  __builtin_mxc_arrive(64 + count);     // global→smem loads
#define arrive_bsmcnt(count)  __builtin_mxc_arrive(4096 + 128*count); // smem→register loads
```

And the combined pipeline barrier:

```cpp
#define ARRIVE_GVM_BSM_BARRIER(gvmcnt, bsmcnt) \
    arrive_gvmcnt(gvmcnt); arrive_bsmcnt(bsmcnt); __builtin_mxc_barrier_inst();
```

Overlap comes from **manually interleaving** LDG / LDS / MMA in one body:

```cpp
MMA_MNK_I(0, 0, 0, 0);
LDG_BSM_A_TILE_STAGE_I(tilek + 1, 0, 0);   // async global load, next stage
MMA_MNK_I(0, 0, 0, 1);
...
ARRIVE_GVM_BSM_BARRIER(2*kLdgNumPerStage*(kStage - 3) + 2, 0);
```

The async global load carries an inline K-bound predicate, so no separate
boundary mask is needed:

```cpp
__builtin_mxc_ldg_b128_bsm_predicator(
    bsm_ldgA + ldg_offs[stage][i], &(gA(rowA[stage][i], colA_rowB, tile)),
    0, true, true, false, true, current_k, K, MACA_ICMP_SLT);
```

**Porting lesson:** a Hopper-style producer/consumer split with mbarriers maps to
*this* — counting outstanding loads with `gvmcnt`/`bsmcnt` and arriving. There is
no warpgroup specialization; one warp does load+compute interleaved.

## Epilogue: no shared memory, direct register→global

`MacaEpilogueNormalDirectStore` declares `struct SharedStorage {};` — the epilogue
uses **no shared memory at all**. There is no `stmatrix`; the "reshuffle" is an
**address remapping** from MMA fragment index to output coordinates:

```cpp
int stg_col = (tid % 16) * 4 + (wave % 2) * 64;
int stg_row_base = lane / 16 * 16 + (wave / 2) * 64;
stg_row[i] = stg_row_base + (i % 4) * 4 + (i / 4);
```

followed by a predicated scatter store:

```cpp
arch::maca_global_store<sizeof(FragmentOutput)>(&(gC(stg_row[i], stg_col)), &(out[i]), stg_mask[i]);
```

Per thread: 16 output elements, `StgType = __NATIVE_VECTOR__(2, int32_t)`,
accumulators enter as `INT4 output[]` (16 × 4-i32 = 256 B/thread).

If you are writing an epilogue and allocating shared memory for staging, you are
doing it differently from the shipped kernels — the direct address-remapped store
is the reference pattern.

## Fragment layout for 16x16x16 f16

`mctlass/arch/mma_sm80.h:2637-2649`:

```cpp
using FragmentA = Array<half_t, 4>;
using FragmentB = Array<half_t, 4>;
using FragmentC = Array<half_t, 4>;
```

Each thread holds **4 half elements per operand = one 64-bit register**, with 4
accumulators. This trait's call is B-first:

```cpp
auto results = __builtin_mxc_mma_16x16x16f16(
    b.to_macahalf4(), a.to_macahalf4(),        // B first (this trait only)
    {c[0].get().to_macahalf(), c[1].get().to_macahalf(), c[2].get().to_macahalf(), c[3].get().to_macahalf()});
```

The order is trait-dependent, not builtin-dependent: `mma_sm80.h` calls the same
builtin A-first at lines 398 and 540. Copy the whole trait, not the call line.

Note the grafting artifact: the trait is declared as `Mma<GemmShape<16,16,16>, 32, half_t, ...>`
— the second template argument is still **32** (NVIDIA warp size) even though the
hardware warp is 64. Read the builtin call, not the trait signature.

## Warp reduction: two idioms

mctlass ships both a shuffle path and a shared-memory path, and the shuffle one
has a trap: the **generic `shfl_down_sync` template is a no-op stub**
(`mctlass/epilogue/threadblock/epilogue.h:69-71`):

```cpp
template <class T>
MCTLASS_DEVICE T shfl_down_sync(unsigned long mask, T var, unsigned int laneDelta, int width = 64) {
  return var;                                 // <-- no-op!
}
```

Only explicit specializations (`int32_t`, `half_t`) actually shuffle. Calling the
generic version silently returns the input unchanged — a wrong-answer bug, not a
crash. If you use mctlass's helper, instantiate it on a supported type.

The idiomatic C500 reduction **avoids shuffles entirely**
(`mctlass/reduction/thread/maca_reduction/maca_reduction.h:50-63`) — shared memory
plus arrival:

```cpp
sdata[tid] = sum_i;       __builtin_mxc_arrive(4096);
sdata[tid] += sdata[tid + 32];  __builtin_mxc_arrive(4096);
...
sdata[tid] += sdata[tid + 1];
```

Note `+32` then descending: this sequence is written for a **64-lane** warp, and
it uses `arrive` as an intra-warp ordering primitive instead of
`__shfl_down_sync`. On C500 this is faster and avoids the stub trap.

## Register budget of the reference kernel

From `maca_mma_multistage_i8_tn_128x128x256.h`:

```cpp
ABType    a[kStage][kLdsNumPerK][4];      // 64 mtreg
ABType    b[kStage][kLdsNumPerK][4];      // 64 mtreg
AccumType accum[MMA_M][MMA_N] = {0};      // 4*4*4 = 64 mtreg
```

~160 registers for operands + accumulators, plus ~16 each for `rowA`/`colB`/`ldg_offs`.
Use this as the budget reference for a hand-rolled GEMM.

## When to use mctlass vs hand-roll

- **Grouped GEMM, int8, or bf16/fp16 attention-shaped work**: use the prebuilt
  mctlassEx kernels (`libmctlassEx.so`). They are C500-native and already at
  library throughput.
- **Custom epilogue or fused op**: subclass the mctlass threadblock and replace
  the epilogue — the epilogue taking `SharedStorage{}` is designed for this.
- **Novel algorithm**: hand-roll with the patterns above as the reference. Expect
  to spend the effort on the load/compute interleave and the address remapped
  store, which is where most of the structure lives.
