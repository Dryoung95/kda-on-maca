#include <cstdio>
__global__ void noop(float* out) { if (threadIdx.x == 0) out[0] = 1.0f; }
int main() {
    float* f; cudaMalloc(&f, sizeof(float));
    for (int w=0; w<20; w++) noop<<<104,64>>>(f);
    cudaDeviceSynchronize();
    printf("=== grid size effect on launch overhead (block=64) ===\n");
    for (int grid : {1, 8, 64, 104, 416, 1664}) {
        cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b);
        cudaEventRecord(a);
        for (int r=0;r<500;r++) noop<<<grid,64>>>(f);
        cudaEventRecord(b); cudaDeviceSynchronize();
        float ms=0; cudaEventElapsedTime(&ms,a,b);
        printf("  grid=%5d: %7.3f us/launch\n", grid, ms*1000/500);
        cudaEventDestroy(a); cudaEventDestroy(b);
    }
    return 0;
}
