---
title: C500 Architecture Facts
tags: [architecture, device-properties, warp64, mctlass, mcflashinfer]
architecture: [c500]
type: hardware
provenance: measured
measured: 2026-09-15
probe: torch.cuda.get_device_properties (in-page snippet)
confidence: high
---

# C500 Architecture Facts (measured, not assumed)

Everything on this page is either read off the device or measured by microbenchmark
on this exact host. Where a fact contradicts the NVIDIA-flavored assumption that
CUDA-trained kernel authors carry, the contradiction is the point of the page.

This knowledge base replaces upstream KDA's `KernelWiki` for this target. Upstream
is Blackwell/Hopper-first (`tcgen05`, TMEM, CLC, `wgmma`, FlashAttention-4); none
of that exists on C500, and porting it would have meant inventing content. This
wiki is built from the SDK headers and from measurement.

## Hardware

| Property | Value | How we know |
|---|---|---|
| Name | MetaX C500 | `torch.cuda.get_device_name(0)` |
| SM count | 104 | device properties |
| Total memory | 16.3 GB (16,341,008,384 B) | device properties |
| Warp size | **64** (not 32) | hardcoded in the compiler: `const int warpSize = 64;` |
| Max threads / SM | 2048 | device properties |
| Registers / SM | 131,072 | device properties |
| Shared mem / block | 64 KB | device properties |
| Shared mem / SM | 64 KB | device properties |
| L2 cache | 8 MB | device properties |
| ISA | `maca-mxc-metax-macahca--xcore1000` | embedded in `lib/mctlassEx_xcore1000.mcfb` |
| Compile arch predefine | `__MACA_ARCH__ == 1000` (C500), `1500` (C600) | mctlass headers |

Warp size is not a query — it is a compile-time constant in
`mxgpu_llvm/lib/clang/19/include/__clang_maca_builtin_vars.h`. Code that assumes
32 is silently wrong at 64 and there is no runtime warning.

## What the toolchain actually exposes

The mctlass port tags architectures with NVIDIA-derived names only
(`Sm50`…`Sm86`). There is **no `xcore1000` arch tag** in the headers; C500 code is
grafted onto the `Sm80` tag and gated on `__MACA_ARCH__ == 1000`.

| NVIDIA feature | C500 status | Evidence |
|---|---|---|
| MMA / tensor cores | ✅ but **16x16xK**, not `m16n8kK` | `__builtin_mxc_mma_16x16x*` |
| `cp.async` (SM80) | ❌ hard-disabled | `copy_sm80.hpp: #if defined(__MACA_ARCH__) && 0` |
| `ldmatrix` (SM75) | ❌ disabled | `CUTE_ARCH_LDSM_SM75_ACTIVATED` off; replaced by `__shfl_down_sync` reshuffles |
| TMA (SM90) | ❌ dead code | `CUTE_ARCH_TMA_SM90_ENABLED` commented out |
| `wgmma` (SM90) | ❌ dead code | `CUTE_ARCH_MMA_SM90A_ENABLED` commented out |
| Thread block clusters (SM90) | ❌ dead code | `CUTE_ARCH_CLUSTER_SM90_ENABLED` commented out |
| `tcgen05` (Blackwell) | ❌ absent entirely | zero hits for `tcgen` in cute/mctlass |
| `mbarrier` | ❌ only inside disabled TMA files | — |

**Effective capability: SM75/SM80-era, plus MACA-native builtins.** Any
optimization strategy in a prompt or playbook that names a Hopper or Blackwell
feature is inapplicable here and must be mapped to a C500 substitute
(see [feature-mapping](../migration/feature-mapping.md)).

### MACA-native replacements

| NVIDIA idiom | C500 substitute |
|---|---|
| `cp.async` | `__builtin_mxc_ldg_b{32,64,128}_bsm[_predicator]` |
| `wgmma` arrive/wait | `__builtin_mxc_arrive`; `arrive_gvmcnt(n)` = `__builtin_mxc_arrive(64+n)`, `arrive_bsmcnt(n)` = `4096+128*n` |
| `ldmatrix` fragment load | `__shfl_down_sync` reshuffle (as mctlass does) |
| `mma.sync m16n8kK` | `__builtin_mxc_mma_16x16x*` — **note B-first arg order `(b, a, c)`** |

## Hardware query surface

Do not assume CUDA's 32-bank shared-memory model. C500 topology is queryable
through `mxkw` (`/opt/maca-3.3.0/include/mxkw/mxkwtypes.h`), which exposes
`WaveFrontSize` ("typically 64"), `WSMSizeInKB`, `NumShaderBanks`, `MaxWavesPerPEU`,
`NumArrays`. `MXKW_GPU_MODEL_MXC500 = 2` is the C500 model enum.

**Caveat:** `mctlass/arch/arch.h`'s `SmId()` is a stub returning uninitialized
memory (its PTX is commented out). Do not use it for SM-affine work — it is not
 merely unimplemented, it returns garbage.

## Reference implementations worth reading

- `include/mcflashinfer/attention/prefill_kernels_xcore1000.cuh` — C500-native
  prefill kernel, the closest thing to an official "how to write a good C500
  kernel" example.
- `include/mcflashinfer/attention/mla_kernels_xcore1000.cuh` — MLA attention,
  `_ctq64` and `_ctq32` variants.
- `include/mcflashinfer/permuted_smem.cuh` — XOR swizzle in
  `SwizzleMode::{k64B,k128B}`; this is the sanctioned bank-conflict-free layout
  pattern (not NVIDIA's `swizzle_layout.hpp`).
- `lib/mctlassEx_xcore1000.mcfb` + `libmctlassEx.so` — prebuilt C500-native
  group GEMM kernels (`Mck_int8_masked_group_gemm_tn_*`).

mcflashinfer ships as headers (compile-in), mctlassEx ships prebuilt for
xcore1000. Both dispatch on `__MACA_ARCH__` and assert on anything else.

## How this page stays honest

Every number above came from either a header the compiler ships (so it is
authoritative for this toolchain version) or a measurement on this board. When a
NVIDIA assumption fails, the page says how it fails and what to do instead. Pages
that depend on measurement link to the probe that produced them, and each carries
the measurement date, because these values can change between SDK releases.
