#include <cstdio>
#include <algorithm>
// Pure LOAD sweep (no store traffic), isolates read-port behavior. The address
// and value both depend on the iteration so the load cannot be hoisted out of
// the loop, which would otherwise make the stride-1 baseline unreproducible.
__global__ void smem_load(float* out, int stride, int niter) {
    __shared__ float smem[4096];
    int tid = threadIdx.x;
    float v = 0;
    for (int it = 0; it < niter; it++) {
        v += smem[(tid * stride + it) & 4095] + it;
    }
    if (v < -1e30f) out[blockIdx.x] = v;   // sink
}
int main() {
    float* f; cudaMalloc(&f, 1024*sizeof(float));
    // Settle the device first: the first sweep in a fresh process runs ~1.6x
    // slower than every later one regardless of stride, which would make any
    // ratio against stride 1 meaningless.
    for (int w = 0; w < 200; w++) smem_load<<<8,256>>>(f, 1, 2000);
    cudaDeviceSynchronize();
    for (int w = 0; w < 200; w++) smem_load<<<8,256>>>(f, 32, 2000);
    cudaDeviceSynchronize();
    // This kernel is short (~8 us), so a single timing window is dominated by
    // clock noise. Take the median of several windows instead of one average.
    printf("=== pure LOAD stride sweep (no syncthreads), median of 5 windows ===\n");
    for (int s = 1; s <= 64; s++) {
        for (int w=0; w<3; w++) smem_load<<<8,256>>>(f, s, 2000);
        cudaDeviceSynchronize();
        double samples[5];
        for (int rep = 0; rep < 5; rep++) {
            cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b);
            cudaEventRecord(a);
            for (int r=0;r<10;r++) smem_load<<<8,256>>>(f, s, 2000);
            cudaEventRecord(b); cudaDeviceSynchronize();
            float ms=0; cudaEventElapsedTime(&ms,a,b);
            samples[rep] = ms*1000/10;
            cudaEventDestroy(a); cudaEventDestroy(b);
        }
        std::sort(samples, samples+5);
        printf("stride=%3d  %8.2f us\n", s, samples[2]);
    }
    return 0;
}
