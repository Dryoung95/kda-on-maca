# Harness Guide

A **profiling harness** is a small standalone CUDA executable whose sole
purpose is to launch the kernel you want to profile, with realistic inputs.
Ported from the upstream ncu-report-skill guide.

You should almost always build a harness when profiling a kernel that lives
inside:

- **PyTorch inline-compiled CUDA** — compiled via
  `torch.utils.cpp_extension.load_inline`, which works on MACA (verified in
  the KernelBench migration) but produces multi-kernel traces.
- **Triton kernels** — Triton's JIT makes it hard to pin a specific compiled
  artifact.
- **CUTLASS / mctlassEx** — layered build system.
- **A larger binary** where the kernel of interest is buried under
  initialization, data loading, or framework code that makes the trace noisy.

You can skip the harness if:

- You specifically want the in-context view — which kernels run before and
  after, how much host-side work separates launches. mcTracer's whole-run
  trace is genuinely better than ncu for this question.
- The kernel already builds cleanly with `cucc` and iterating is fast enough.

---

## What a good harness contains

1. The kernel (verbatim copy of the device code + any `__device__` helpers).
2. Explicit template instantiations for every template-parameter combination
   you plan to profile (e.g. `<TILE_M, TILE_N>`, `<VEC_WIDTH, BLOCK_SIZE>`).
3. Optional input loading — from a binary file or synthetic.
4. A minimal `main()` that parses CLI args, allocates GPU memory, launches
   the kernel, synchronizes, and exits.

Things that should NOT be in the harness:

- **Framework dependencies (torch, pybind11)** — they slow the build and add
  dozens of unrelated kernels to the trace.
- **Multi-kernel pipelines** — profile each kernel separately unless
  measuring kernel-to-kernel interactions.
- **Correctness checks** — verify correctness separately (the KernelBench
  harness exists for exactly this), don't couple it to profiling.
- **Long timing loops** — unlike ncu, mcTracer has no replay, so one launch
  after a warm-up is enough. A warm-up launch matters: the first launch pays
  module-load and context-warmup cost.

---

## Template

A complete reusable template lives at
[`../helpers/harness_template.cu`](../helpers/harness_template.cu). Customize:

1. **Replace `KERNEL_INCLUDE_GOES_HERE`** with `#include` or paste the kernel.
2. **Add explicit instantiations** for every template parameter combination.
3. **Define the input shape parameters** (grid/block sizes, tensor shapes).
4. **Fill in `alloc_and_fill()`** to allocate/initialize inputs.
5. **Fill in `launch_kernel()`** to do the actual launch.

Compile with:

```bash
source /path/to/kda-maca/env.sh
cucc my_kernel_harness.cu -O2 -std=c++17 $MACA_CUCC_FLAGS -o my_kernel_harness
```

`$MACA_CUCC_FLAGS` expands to `-DUSE_MACA -I<cu-bridge>/include`. It is
required — see [`12-common-issues.md`](12-common-issues.md).

---

## Real data vs synthetic data

Three levels of fidelity for harness inputs:

### Level 1: Arbitrary synthetic

`float* x = cudaMalloc(...)` without initialization.

**Use when:** you only care about shape-dependent perf and the kernel has no
data-dependent control flow. Fine for matmul-shaped kernels.

**Don't use when:** the kernel branches on input values (early-exit on a
threshold, histogram with real distribution), or when timing depends on
denormal-flush / NaN handling.

### Level 2: Patterned synthetic

Fill on the host with a cheap pattern, or `cudaMemset` to a constant.

**Use when:** you need deterministic-but-representative inputs and real data
is inconvenient. Good for element-wise and reduction kernels.

### Level 3: Real data

Load real workload tensors. On MACA the practical route is to dump tensors
from a PyTorch run to a raw binary (or safetensors — the `safetensors` wheel
is installed) and `fread` them in the harness.

**Use when:** the kernel's behavior depends on the actual input distribution
— variable sequence lengths, sparse patterns, skewed workloads. This is the
only level that exposes tail effects (upstream Pattern B).

---

## Which level do I need?

| Kernel type | Minimum level |
|---|---|
| Dense GEMM / matmul | Level 1 — shape determines everything |
| Element-wise | Level 1 |
| Reduction | Level 1 |
| Attention with variable seq lens | **Level 3** — tail effect is the point |
| Sparse / gather-scatter | Level 3 |
| Histogram / atomic-heavy | Level 3 — input distribution drives conflict rate |

When in doubt, use Level 2. It costs one extra `cudaMemset` and removes the
"was that just uninitialized memory noise?" question.
