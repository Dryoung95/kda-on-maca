#include <cstdio>
#include <cuda_runtime.h>

// Correct warp-64 sum reduction: 6 rounds.
__device__ float reduce6(float v) {
    unsigned long long mask = 0xffffffffffffffffULL;
    float acc = v;
    for (int i = 32; i > 0; i >>= 1)
        acc += __shfl_down_sync(mask, acc, i);
    return acc;
}

// NVIDIA-style 5-round reduction, correct only for warpSize=32.
__device__ float reduce5(float v) {
    unsigned long long mask = 0xffffffffffffffffULL;
    float acc = v;
    for (int i = 16; i > 0; i >>= 1)
        acc += __shfl_down_sync(mask, acc, i);
    return acc;
}

__global__ void k(float* out) {
    // lane i holds value i, so sum = 0+1+...+63 = 2016
    float v = (float)threadIdx.x;
    out[0] = reduce5(v);
    out[1] = reduce6(v);
}

int main() {
    float* d;
    cudaMallocManaged(&d, 2 * sizeof(float));
    d[0] = d[1] = 0;
    k<<<1, 64>>>(d);
    cudaDeviceSynchronize();
    printf("5-round: %.1f   6-round: %.1f   expected: 2016.0\n", d[0], d[1]);
    return 0;
}
