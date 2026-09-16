# MMA / Tensor Cores on C500

Tensor cores exist on C500 and they are the dominant performance axis for
matrix-shaped work. They also differ from NVIDIA's in the one way that matters
when writing or porting a kernel: **the tile shape and the operand order.**

## Shape: 16x16xK, not m16n8kK

NVIDIA's `mma.sync` family (Volta through Ampere) uses `m16n8kK` fragments.
C500's native MMA is **16x16 in M and N**, per the builtins the compiler exposes:

```
__builtin_mxc_mma_16x16x4f64
__builtin_mxc_mma_16x16x4f32
__builtin_mxc_mma_16x16x8tf32
__builtin_mxc_mma_16x16x16f16
__builtin_mxc_mma_16x16x16bf16
__builtin_mxc_mma_16x16x16i8
```

These six are the set evidenced in `include/mctlass/arch/`. The wider/clamped
int8 variants (`16x16x32i8`, `*_clamp`) are listed in some SDK docs but no call
site exists in the headers shipped with 3.3.0.15 — verify them against your own
SDK version before relying on them.

Every shape is 16x16 in M/N. mctlass exposes these as `GemmShape`s `16,16,4`
(f32), `16,16,8` (tf32/f32), `16,16,16` (f16/bf16/i8), plus `8,8,4` for f64.

## Operand order: B first — sometimes

```cpp
// NVIDIA: mma.sync with (a, b, c)
// C500:   __builtin_mxc_mma_16x16x16f16(b, a, c)     <-- B first
```

**The order is not uniform across the mctlass traits.** In
`mctlass/arch/mma_sm80.h` alone, the `16x16x16f16` builtin is called B-first at
lines 2660 and 2722 but A-first at lines 398 and 540. Which one you get depends
on which trait you instantiate, not on the builtin.

mctlass reconciles this by swapping its operands upstream of the call, so
reading mctlass code and copying the *whole* call verbatim is safe; reading
NVIDIA code and copying is not, and copying a single mctlass builtin call in
isolation is not either. A/B transposition is a silent correctness bug — it
produces a valid matrix, just the wrong one, which passes shape checks and fails
numerical comparison.

## Why the tensor-core gap is worth chasing

Measured on this board (4096² matmul, scalar elementwise FMA for comparison):

| Path | Throughput | vs scalar |
|---|---:|---:|
| Library GEMM (tf32 MMA) | 87.5 TFLOPS | **506×** |
| Scalar FMA elementwise | 0.173 TFLOPS | 1× |

A matmul-shaped kernel written with scalar FMA is leaving two orders of magnitude
on the table. This is the single largest optimization lever on the board — larger
than any memory or scheduling improvement — and it applies to GEMM, attention,
and conv-shaped work.

Note what the scalar number means for diagnosis: 0.173 TFLOPS is *below* the
memory ceiling of 1.49 TB/s for any arithmetic intensity above ~0.12 FLOPs/byte,
so a scalar-FMA kernel is not compute-bound, it is ALU-latency-bound. The fix is
not better memory access; it is MMA.

## How to get there

- **Use the library first.** mcBLAS GEMM kernels (`mcblas__Mck_tf32gemm_*` in an
  mcTracer trace) are C500-native and already tuned. Before hand-rolling, check
  whether the operation can be expressed as a library call.
- **mctlassEx** ships prebuilt C500 group-GEMM kernels
  (`lib/mctlassEx_xcore1000.mcfb`, `Mck_int8_masked_group_gemm_tn_*`) for
  grouped/int8 work.
- **Hand-rolling**: use `__builtin_mxc_mma_16x16x*` directly, or mctlass
  templates gated on `__MACA_ARCH__ == 1000`. The mctlass path handles fragment
  layout; the builtin path requires you to manage it.
- **Do not** use NVIDIA `mma.sync` inline asm. It is not the MACA path and the
  atom will not map.

## Fragment loading

`ldmatrix` is disabled on C500 (`CUTE_ARCH_LDSM_SM75_ACTIVATED` off). mctlass
loads MMA fragments with `__shfl_down_sync`-based reshuffles instead — see
`mctlass` `mma_sm80.hpp` around the fragment load path. Any hand-rolled fragment
loader that assumes `ldmatrix` semantics needs replacing.

## What does not exist

No `wgmma` (Hopper), no `tcgen05` (Blackwell), no TMEM accumulators, no TMA
descriptor-based bulk copies. Async data movement is
`__builtin_mxc_ldg_b{32,64,128}_bsm[_predicator]`, and arrival tracking is
`__builtin_mxc_arrive` with the encodings `arrive_gvmcnt(n)` = `64+n` and
`arrive_bsmcnt(n)` = `4096+128*n` (from `maca_kernel_utils.hpp`).

The practical consequence: **an optimization playbook written for Hopper or
Blackwell is not portable here.** Strategies named after those features must be
translated to SM80-era equivalents plus the MACA async builtins.

## Reference implementations

The shipped C500-native attention kernels are the best examples of idiomatic
fragment and shared-memory management on this hardware:

- `include/mcflashinfer/attention/prefill_kernels_xcore1000.cuh`
- `include/mcflashinfer/attention/mla_kernels_xcore1000.cuh` (`_ctq64`, `_ctq32`)
- `include/mcflashinfer/permuted_smem.cuh` (swizzle patterns)

Read one before writing a tiled kernel from scratch.
