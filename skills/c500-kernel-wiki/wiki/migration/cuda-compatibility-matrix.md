# CUDA → C500 Compatibility Matrix

What a CUDA-trained kernel author can keep, and what must change. Each row was
verified by compiling and running a probe on this board (2026-09-15, MACA SDK
3.3.0.15, cucc through cu-bridge). This is the page to check before porting a
kernel, and the page to update when a probe result changes.

## Works as-is

| Feature | Status | Notes |
|---|---|---|
| `<<<grid, block>>>` launch | ✅ | dim3 grids, all dimensions |
| `__shared__`, `__syncthreads` | ✅ | 64 KB / block |
| Dynamic shared memory (`extern __shared__`) | ✅ | launch with 3rd arg |
| `__shfl_sync` / `__shfl_down_sync` | ✅ | **mask must be 64-bit** |
| `__ballot_sync` | ✅ | **returns 64-bit** |
| `atomicAdd` and family | ✅ | |
| `__fmaf_rn` | ✅ | |
| `float4` / vector loads and stores | ✅ | |
| `__half` arithmetic (`__hadd` etc.) | ✅ | `cuda_fp16.h` |
| `__nv_bfloat16` | ✅ | `cuda_bf16.h` |
| Fast math: `__expf` `__logf` `__sinf` `sqrtf` | ✅ | |
| `__launch_bounds__` | ✅ | respected for register budgeting |
| `threadIdx` / `blockIdx` / `blockDim` / `gridDim` | ✅ | |
| Grid-stride loops | ✅ | |
| Dynamic parallelism (device-side launch) | ✅ | verified by SDK sample |
| CUDA graphs / stream capture | ✅ | `mcStreamCaptureModeGlobal` |
| Unified memory (`cudaMallocManaged`) | ✅ | no prefetch/advise hints |
| Stream-ordered allocation (`cudaMallocAsync`) | ✅ | |
| IPC handles | ✅ | requires unified addressing |

The common through-line: **CUDA surface syntax ports cleanly**. cucc accepts
CUDA-syntax source and the compatibility layer handles the rest. The KernelBench
migration confirmed 182/250 problems run with zero source modifications, so the
default assumption for any CUDA kernel should be "it probably compiles" — and
the exceptions below are where that assumption breaks.

## Broken or different

| Feature | Status | What happens | Fix |
|---|---|---|---|
| **`cooperative_groups::grid_group::sync()`** | ❌ | `g.is_valid()` returns false; calling `sync()` triggers a device assert that disables the runtime | Do not use grid-level sync. Use atomic-counter work queues or split into multiple kernels. |
| Warp size 32 assumptions | ❌ silent | `warpSize` is 64; 5-round warp reductions return half-reduced results | Derive rounds from `warpSize`; see [warp64-implications](../hardware/warp64-implications.md) |
| 32-bit `__shfl` masks | ❌ silent | `0xffffffff` leaves top 32 lanes out | `0xffffffffffffffffULL` |
| `cp.async` | ❌ | hard-disabled in cute (`#if defined(__MACA_ARCH__) && 0`) | `__builtin_mxc_ldg_b{32,64,128}_bsm` |
| `ldmatrix` | ❌ | disabled in cute | `__shfl_down_sync` fragment reshuffle (as mctlass does) |
| `mma.sync m16n8kK` | ❌ shape | C500 MMA is **16x16xK**, B-first | `__builtin_mxc_mma_16x16x*`; see [mma-and-tensor-cores](../hardware/mma-and-tensor-cores.md) |
| TMA / `wgmma` / clusters / `tcgen05` | ❌ | present as dead code, gated off | no equivalent; use SM80-era strategies |

The grid-group failure deserves emphasis because it fails *silently at first*:
`is_valid()` returns false, so a guarded kernel quietly skips the sync and
produces wrong results, and an unguarded kernel asserts and disables the MACA
runtime for the rest of the process. Neither failure mode produces a helpful
error. Any kernel pattern that relies on cross-block synchronization on-device
needs restructuring.

## The silent-failure class

Three of the rows above — warp size, shfl mask width, grid sync — share a
dangerous property: **they do not raise errors**. The kernel compiles, launches,
and returns wrong numbers. On NVIDIA these are correct-by-construction; on C500
they are correct-by-coincidence only for symmetric or lucky inputs.

When porting a kernel, audit for these three *before* running it, because a
correctness gate that passes on the default seed can still hide a warp-reduction
bug that only surfaces on asymmetric data.

## Environment-level compatibility

Source compatibility is not the whole story. The following are provided by the
environment, not by the kernel source, and are required for builds to work:

- `cucc` needs `-I$MACA_PATH/tools/cu-bridge/include` or it fails on
  `__macro_mxcc.h`. This is the most common porting failure and it is an
  environment issue, not a source issue.
- `LD_LIBRARY_PATH` must include the torch lib dir when loading a compiled
  extension, or `import` of the extension fails on `libc10.so`.
- `PYTHONPATH` must include the workspace if a model file imports a sibling
  module (KernelBench loads candidates in an isolated temp context).

These are documented in the migration projects, not the kernel source, because
they are properties of how the toolchain is wired on this host.

## Probes

- `probes/probe_compat.cu` — shfl/ballot/dynamic-shmem/fma/atomic
- `probes/probe_compat2.cu` — vector/half/math/launch-bounds/cooperative-groups
- `probes/probe_cg2.cu` — the grid-group `is_valid()` diagnosis
