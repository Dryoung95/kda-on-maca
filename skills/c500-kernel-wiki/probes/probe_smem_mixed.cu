#include <cstdio>
__global__ void smem_stride(float* out, int stride, int niter) {
    __shared__ float smem[4096];
    int tid = threadIdx.x;
    float v = 0;
    for (int it = 0; it < niter; it++) {
        smem[(tid * stride + it) & 4095] = (float)tid + it;
        __syncthreads();
        v += smem[(tid * stride + it) & 4095];
        __syncthreads();
    }
    if (v < -1e30f) out[blockIdx.x] = v;
}
int main() {
    float* f; cudaMalloc(&f, 1024*sizeof(float));
    // Warm up the stride-1 case: its baseline is bimodal until the kernel has
    // run enough times (73.7 vs 186.0 us observed for the store probe).
    // Without this the ratio column below is not reproducible.
    // Settle the device before measuring. The first sweep in a fresh process
    // runs ~1.6x slower than every later one regardless of stride, which
    // makes any ratio involving stride 1 unreproducible without this.
    for (int w = 0; w < 200; w++) smem_stride<<<8,256>>>(f, 1, 200);
    cudaDeviceSynchronize();
    for (int w = 0; w < 200; w++) smem_stride<<<8,256>>>(f, 32, 200);
    cudaDeviceSynchronize();
    printf("=== fine stride sweep (256 thr, mask 4095) ===\n");
    for (int s = 1; s <= 96; s++) {
        for (int w=0; w<3; w++) smem_stride<<<8,256>>>(f, s, 200);
        cudaDeviceSynchronize();
        cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b);
        cudaEventRecord(a);
        for (int r=0;r<20;r++) smem_stride<<<8,256>>>(f, s, 200);
        cudaEventRecord(b); cudaDeviceSynchronize();
        float ms=0; cudaEventElapsedTime(&ms,a,b);
        printf("  stride=%3d  %8.2f us\n", s, ms*1000/20);
        cudaEventDestroy(a); cudaEventDestroy(b);
    }
    return 0;
}
