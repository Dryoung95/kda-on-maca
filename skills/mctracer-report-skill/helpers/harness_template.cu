// Standalone profiling harness template (MACA / cu-bridge).
//
// Port of the upstream ncu-report-skill harness_template.cu. Same purpose:
// compile the kernel in isolation so the trace contains exactly one
// interesting launch, with inputs you control.
//
// How to use:
//   1. Copy this file into profile/<run_name>/harness/
//   2. Paste your kernel (and any __device__ helpers) below, or #include it.
//   3. Fill in alloc_and_fill() and launch_kernel().
//   4. Build:
//        source <kda-maca>/env.sh
//        cucc my_kernel_harness.cu -O2 -std=c++17 $MACA_CUCC_FLAGS -o my_kernel_harness
//   5. Trace:
//        mcTracer --odname ../reports --name <tag> ./my_kernel_harness [args]
//
// There is no -lineinfo equivalent to set: mcTracer is an API-level tracer
// and does not consume debug info. Kernel names in the trace come from the
// symbol table, so keep the kernel extern in this translation unit.

#include <cstdio>
#include <cstdlib>
#include <cmath>

// --- KERNEL_INCLUDE_GOES_HERE ------------------------------------------------
// Paste the kernel here, e.g.:
//
// __global__ void saxpy_kernel(int n, float a, const float* x, float* y) {
//     int i = blockIdx.x * blockDim.x + threadIdx.x;
//     if (i < n) y[i] = a * x[i] + y[i];
// }

// --- alloc_and_fill ----------------------------------------------------------
// Allocate and initialize inputs for the kernel. Three fidelity levels:
//   Level 1 (synthetic): cudaMalloc + cudaMemset, shape-only perf.
//   Level 2 (patterned) : fill with a cheap host-side pattern.
//   Level 3 (real data) : load real tensors — see reference/02-harness-guide.md.
static void alloc_and_fill(/* TODO(you): out-params */) {
    // TODO(you)
}

// --- launch_kernel -----------------------------------------------------------
static void launch_kernel(/* TODO(you): inputs */) {
    // TODO(you): dim3 grid/block, kernel<<<grid, block>>>(...)
}

int main(int argc, char** argv) {
    // Optional CLI shape overrides: ./harness M N K
    (void)argc; (void)argv;

    alloc_and_fill();

    // Warm-up launch: mcTracer records every launch, and the first one pays
    // module-load and context-warmup cost. Keep the warm-up, then trace the
    // steady-state launch below.
    launch_kernel();
    cudaDeviceSynchronize();

    // The interesting launch — exactly one, so the trace stays readable.
    // ncu used -c 1 for the same reason; mcTracer has no replay, so instead
    // we simply launch once more after warm-up.
    launch_kernel();
    cudaDeviceSynchronize();

    // Correctness is checked separately from profiling — do it in a
    // KernelBench run, not here.
    printf("harness done\n");
    return 0;
}
