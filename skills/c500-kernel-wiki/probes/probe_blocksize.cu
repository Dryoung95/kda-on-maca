#include <cstdio>
__global__ void probe_granularity(float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = 1.0f;
}
int main() {
    int n = 1<<20;
    float* f; cudaMalloc(&f, n*sizeof(float));
    printf("=== block-size sweep (1M elements, 50 reps, cudaEvent) ===\n");
    for (int bs = 32; bs <= 1024; bs *= 2) {
        int grid = (n + bs - 1)/bs;
        for (int w=0; w<5; w++) probe_granularity<<<grid,bs>>>(f, n);
        cudaDeviceSynchronize();
        cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b);
        cudaEventRecord(a);
        for (int r=0;r<50;r++) probe_granularity<<<grid,bs>>>(f, n);
        cudaEventRecord(b); cudaDeviceSynchronize();
        float ms = 0; cudaEventElapsedTime(&ms,a,b);
        printf("  block=%4d grid=%6d  %8.2f us/call\n", bs, grid, ms*1000/50);
        cudaEventDestroy(a); cudaEventDestroy(b);
    }
    cudaFree(f);
    return 0;
}
