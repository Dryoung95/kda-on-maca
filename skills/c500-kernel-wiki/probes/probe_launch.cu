#include <cstdio>
__global__ void noop(float* out) { if (threadIdx.x == 0) out[0] = 1.0f; }
int main() {
    float* f; cudaMalloc(&f, sizeof(float));
    for (int w=0; w<20; w++) noop<<<1,1>>>(f);
    cudaDeviceSynchronize();
    printf("=== launch overhead: empty kernel <<<1,1>>> ===\n");
    for (int bs : {1, 64, 256}) {
        cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b);
        cudaEventRecord(a);
        for (int r=0;r<1000;r++) noop<<<1,bs>>>(f);
        cudaEventRecord(b); cudaDeviceSynchronize();
        float ms=0; cudaEventElapsedTime(&ms,a,b);
        printf("  block=%3d: %7.3f us/launch\n", bs, ms*1000/1000);
        cudaEventDestroy(a); cudaEventDestroy(b);
    }
    return 0;
}
