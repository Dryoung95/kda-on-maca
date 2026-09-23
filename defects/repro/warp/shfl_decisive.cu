#include <cstdio>
#include <cuda_runtime.h>

// Decisive probe of MACA __shfl_down_sync semantics on a 64-lane warp.
// Question: for offset >= 32, is the warp treated as one 64-lane warp
// (CUDA semantics: out-of-range source -> own value) or as two 32-lane
// halves with data copied between them?
//
// Discriminating experiment: set lane values so that a "copy" and a
// "no-op" give different answers. lane i gets v = 1000 + i. Then for
// offset 32, lane 0: copy-from-high-half predicts 1032; CUDA-noop
// predicts 1000. Lane 32: copy-from-low-half predicts 1000; CUDA-noop
// predicts 1032.
__global__ void probe(int* out) {
    int lane = threadIdx.x;
    int v = 1000 + lane;

    int down32 = __shfl_down_sync(0xffffffffffffffffULL, v, 32);
    int down48 = __shfl_down_sync(0xffffffffffffffffULL, v, 48);
    int down63 = __shfl_down_sync(0xffffffffffffffffULL, v, 63);
    int up32   = __shfl_up_sync(0xffffffffffffffffULL, v, 32);
    int idx32  = __shfl_sync(0xffffffffffffffffULL, v, 32);

    if (lane == 0 || lane == 1 || lane == 31 || lane == 32 || lane == 33 || lane == 63) {
        printf("lane %2d: v=%d  d32=%d  d48=%d  d63=%d  u32=%d  idx32=%d\n",
               lane, v, down32, down48, down63, up32, idx32);
    }
    out[lane] = down32;
}

// Full-warp reduction over all 64 lanes, sum of lane ids = 2016.
// Variant A: 6 rounds starting at offset 32 (NVIDIA style for warp 64).
// Variant B: 5 rounds starting at offset 16.
// Variant C: tree with __shfl_sync to an explicit lane.
__device__ int reduce6(int v) {
    int a = v;
    for (int i = 32; i > 0; i >>= 1) a += __shfl_down_sync(0xffffffffffffffffULL, a, i);
    return a;
}
__device__ int reduce5(int v) {
    int a = v;
    for (int i = 16; i > 0; i >>= 1) a += __shfl_down_sync(0xffffffffffffffffULL, a, i);
    return a;
}
__device__ int reduce_sync(int v) {
    int a = v;
    for (int i = 32; i > 0; i >>= 1) a += __shfl_sync(0xffffffffffffffffULL, a, (threadIdx.x + i) % 64);
    return a;
}

__global__ void red(int* out) {
    int v = threadIdx.x;
    out[0] = reduce5(v);
    out[1] = reduce6(v);
    out[2] = reduce_sync(v);
}

int main() {
    int *d;
    cudaMallocManaged(&d, 64 * sizeof(int));
    probe<<<1, 64>>>(d);
    cudaDeviceSynchronize();
    printf("\n--- reductions, correct answer 2016 ---\n");
    red<<<1, 64>>>(d);
    cudaDeviceSynchronize();
    printf("5-round shfl_down: %d\n", d[0]);
    printf("6-round shfl_down: %d\n", d[1]);
    printf("6-round shfl_sync: %d\n", d[2]);
    cudaFree(d);
    return 0;
}
