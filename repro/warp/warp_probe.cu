#include <cstdio>
#include <cuda_runtime.h>

// Probe: what does __shfl_down_sync(mask, v, 32) actually do on C500?
// CUDA semantics on warp 32: offset clamps to warpSize-1, so offset>=32 is a no-op.
// If MACA treats the 64-lane warp as two 32-lane halves, offset 32 might
// move data between halves (that is the assumption behind "6 rounds").
__global__ void probe(float* out) {
    int lane = threadIdx.x;
    float v = (float)lane;

    float down16 = __shfl_down_sync(0xffffffffffffffffULL, v, 16);
    float down32 = __shfl_down_sync(0xffffffffffffffffULL, v, 32);
    float up32   = __shfl_up_sync(0xffffffffffffffffULL, v, 32);

    if (lane < 4 || (lane >= 32 && lane < 36)) {
        printf("lane %2d: v=%g  down16=%g  down32=%g  up32=%g\n",
               lane, v, down16, down32, up32);
    }
    out[0] = down32;
}

int main() {
    float* d;
    cudaMallocManaged(&d, sizeof(float));
    probe<<<1, 64>>>(d);
    cudaDeviceSynchronize();
    return 0;
}
