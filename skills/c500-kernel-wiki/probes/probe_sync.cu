#include <cstdio>
__global__ void sync_only(float* out, int nsync) {
    int tid = threadIdx.x;
    volatile float v = tid;
    for (int it = 0; it < 200; it++) {
        for (int k = 0; k < nsync; k++) __syncthreads();
        v += it;
    }
    if (v < -1e30f) out[blockIdx.x] = v;
}
int main() {
    float* f; cudaMalloc(&f, 1024*sizeof(float));
    printf("=== __syncthreads cost per call (256 threads, 200 outer iters) ===\n");
    for (int ns : {0, 1, 2, 4, 8}) {
        for (int w=0; w<3; w++) sync_only<<<8,256>>>(f, ns);
        cudaDeviceSynchronize();
        cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b);
        cudaEventRecord(a);
        for (int r=0;r<10;r++) sync_only<<<8,256>>>(f, ns);
        cudaEventRecord(b); cudaDeviceSynchronize();
        float ms=0; cudaEventElapsedTime(&ms,a,b);
        double per_sync = ns ? (ms*1000/10/200/ns) : 0;
        printf("  syncs/iter=%d  %8.2f us   per-sync=%.4f us\n", ns, ms*1000/10, per_sync);
        cudaEventDestroy(a); cudaEventDestroy(b);
    }
    return 0;
}
